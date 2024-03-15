import json
import logging
import os
import random
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Union

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from nn_core.common import PROJECT_ROOT
from sentence_aligner.data.dataset import CustomLoader, MyDataset
from sentence_aligner.util.utils import GlossManager, char2pos, load_inventory

pylogger = logging.getLogger(__name__)


class CustomSample(Dict):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def __len__(self):
        return self.size


class MetaData:
    def __init__(self, dataset_names: Mapping[str, List[str]]):
        """The data information the Lightning Module will be provided with.

        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.

        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.

        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.

        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.

        Args:
            class_vocab: association between class names and their indices
        """
        # example
        self.dataset_names: Mapping[str, List[str]] = dataset_names

    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        # example
        (dst_path / "class_vocab.json").write_text(json.dumps(self.dataset_names))

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint

        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        dataset_names = json.loads(
            (src_path / "class_vocab.json").read_text(encoding="utf-8")
        )

        return MetaData(
            dataset_names=dataset_names,
        )


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


def encode_samples(
    samples: List,
    context_tokenizer: PreTrainedTokenizer,
    gloss_tokenizer: PreTrainedTokenizer,
):
    pretokenized = any([a["pretokenized"] for a in samples if "pretokenized" in a])

    samples_sources = [s["sources"] for s in samples]
    samples_targets = [s["targets"] for s in samples]
    sources_ids = [g for s in samples for g in s["sources_ids"]]
    targets_ids = [g for s in samples for g in s["targets_ids"]]

    sources = [g for s in samples for g in s["sources"]]
    targets = [g for s in samples for g in s["targets"]]
    sources_encodings = gloss_tokenizer(
        sources,
        truncation=True,
        padding=True,
        return_tensors="pt",
        is_split_into_words=pretokenized,
        # max_length=100,
        return_token_type_ids=True,
    )
    source_sizes = [len(s["sources"]) for s in samples]
    target_sizes = [len(s["targets"]) for s in samples]
    i = 0
    j = 0
    gold_indices = set()
    for source_sents, target_sents in zip(samples_sources, samples_targets):
        source_indices, target_indices = [], []
        for i_i, source_sent in enumerate(source_sents):
            source_indices.append(i_i)
        for j_i, target_sent in enumerate(target_sents):
            target_indices.append(j_i)
        for i_i in source_indices:
            for j_i in target_indices:
                gold_indices.add((i_i + i, j_i + j))
        i += len(source_indices)
        j += len(target_indices)
    gold_matrix = []
    for i, j in torch.cartesian_prod(
        torch.arange(0, len(sources)), torch.arange(0, len(targets))
    ).tolist():
        gold_matrix.append((i, j) in gold_indices)

    target_encodings = context_tokenizer(
        targets,
        is_split_into_words=pretokenized,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
        return_token_type_ids=True,
    )

    sources_embeds = torch.stack([g for s in samples for g in s["sources_embs"]])
    targets_embeds = torch.stack([g for s in samples for g in s["targets_embs"]])

    output = dict()
    output["sources_text"] = sources
    output["targets_text"] = targets
    output["sources_ids"] = sources_ids
    output["targets_ids"] = targets_ids
    output["sources_encodings"] = sources_encodings
    output["targets_encodings"] = target_encodings
    output["sources_embeds"] = sources_embeds
    output["targets_embeds"] = targets_embeds
    output["source_sizes"] = source_sizes
    output["target_sizes"] = target_sizes
    output["gold_matrix"] = torch.tensor(gold_matrix, dtype=torch.long)

    return output


def collate_fn(
    samples: List,
    gloss_tokenizer: PreTrainedTokenizer,
    context_tokenizer: PreTrainedTokenizer,
):
    encoded_example = encode_samples(
        samples, gloss_tokenizer=gloss_tokenizer, context_tokenizer=context_tokenizer
    )
    return encoded_example


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        root_path: str,
        transformer_name: str,
        val_dataset_names: List[str] = None,
        train_dataset_names: Optional[List[str]] = None,
        test_dataset_names: Optional[List[str]] = None,
        max_ids: int = 2000,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#gpus
        self.pin_memory: bool = gpus is not None and str(gpus) != "0"

        self.max_ids = max_ids
        self.root_path: Path = Path(root_path)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            transformer_name
        )
        self.train_dataset: Optional[MyDataset] = None
        self.val_datasets: Optional[Sequence[MyDataset]] = None
        self.test_datasets: Optional[Sequence[MyDataset]] = None

        self.split_types2names: Dict[str, Optional[Set[str]]] = dict(
            train=None if train_dataset_names is None else set(train_dataset_names),
            val=None if val_dataset_names is None else set(val_dataset_names),
            test=None if test_dataset_names is None else set(test_dataset_names),
        )
        self.samples: Dict[str, Dict[str, Dict[str, Any]]] = self._read_data()

    def _read_data(self):
        sample_data = dict()

        r = random.Random(42)
        for split in ["train", "val", "test"]:
            valid_dataset_names: Optional[Set[str]] = self.split_types2names[split]

            data_path: Path = self.root_path / split
            for dataset_file in os.listdir(data_path):
                if (valid_dataset_names is not None) and (
                    dataset_file not in valid_dataset_names
                ):
                    continue

                if split == "train":
                    sample_data.setdefault(split, dict()).setdefault(
                        "train", []
                    ).append(data_path / dataset_file)
                else:
                    sample_data.setdefault(split, dict())[dataset_file] = [
                        data_path / dataset_file
                    ]
        return sample_data

    @cached_property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        # Since MetaData depends on the training data, we need to ensure the setup method has been called.
        if self.train_dataset is None:
            self.setup(stage="fit")

        dataset_names = dict()

        for dataset in self.train_dataset:
            dataset_names.setdefault("train", []).append(dataset.task_name)
        for dataset in self.val_datasets:
            dataset_names.setdefault("val", []).append(dataset.task_name)
        if self.test_datasets is not None:
            for dataset in self.test_datasets:
                dataset_names.setdefault("test", []).append(dataset.task_name)
        return MetaData(dataset_names)

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.

        if stage is None or stage == "fit":
            self.train_dataset = [
                hydra.utils.instantiate(
                    self.datasets.train, datamodule=self, task_name=task_name
                )
                for task_name in self.samples["train"].keys()
            ]
            self.val_datasets = [
                hydra.utils.instantiate(
                    self.datasets.val, datamodule=self, task_name=task_name
                )
                for task_name in self.samples["val"].keys()
            ]
            self.test_datasets = [
                hydra.utils.instantiate(
                    self.datasets.test, datamodule=self, task_name=task_name
                )
                for task_name in self.samples["test"].keys()
            ]

        # if stage is None or stage == "test":

    def train_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                # max_ids=self.max_ids,
                # shuffle=True,
                batch_size=self.batch_size.train,
                num_workers=self.num_workers.train,
                worker_init_fn=worker_init_fn,
                pin_memory=self.pin_memory,
                collate_fn=partial(
                    collate_fn,
                    gloss_tokenizer=self.tokenizer,
                    context_tokenizer=self.tokenizer,
                ),
            )
            for dataset in self.train_dataset
        ]

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
                pin_memory=self.pin_memory,
                collate_fn=partial(
                    collate_fn,
                    gloss_tokenizer=self.tokenizer,
                    context_tokenizer=self.tokenizer,
                ),
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                pin_memory=self.pin_memory,
                worker_init_fn=worker_init_fn,
                collate_fn=partial(
                    collate_fn,
                    gloss_tokenizer=self.tokenizer,
                    context_tokenizer=self.tokenizer,
                ),
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    _: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)


if __name__ == "__main__":
    main()
