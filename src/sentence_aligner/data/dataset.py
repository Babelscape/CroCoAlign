import json
from typing import Dict, Iterator, Tuple, Union

import hydra
import omegaconf
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision.datasets import FashionMNIST

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split


class CustomLoader(DataLoader):
    def __init__(self, dataset, max_ids: int, **kwargs):
        super().__init__(dataset, **kwargs)
        self.ds = dataset
        self.max_ids = max_ids

    def __iter__(self):
        batch = []
        batch_size = 0
        for sample in self.ds:
            if batch_size + len(sample) < self.max_ids:
                batch.append(sample)
                batch_size += len(sample)
                continue
            elif len(batch) == 0:
                continue
            to_return_batch = batch
            batch = []
            batch_size = 0
            yield self.collate_fn(to_return_batch)
        if len(batch) > 0:
            yield self.collate_fn(batch)

def sample_generator(file_paths):
    for dataset_file in file_paths:
        with (dataset_file).open("r") as fr:
            for line in fr:
                # if len(samples) > 10000:
                #     break
                example = json.loads(line)
                yield dict(sources=example["sources"]["text"],
                            sources_ids=example["sources"]["ids"],
                            sources_embs=torch.tensor(example["sources"]["embs"]),
                            targets=example["targets"]["text"],
                            targets_ids=example["targets"]["ids"],
                            targets_embs=torch.tensor(example["targets"]["embs"]))

class MyDataset(IterableDataset):
    def __init__(self, split: Split, datamodule, task_name, **kwargs):
        super().__init__()
        self.split: Split = split
        self.task_name = task_name
        self.file_paths = datamodule.samples[split][task_name]

    def __iter__(self) -> Iterator:
        return sample_generator(self.file_paths)

    # def __len__(self) -> int:
    #     return len(self.samples)

    # def __getitem__(self, index) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    #     return self.samples[index]

    def __repr__(self) -> str:
        return f"MyDataset({self.split=}, n_instances={len(self)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    _: Dataset = hydra.utils.instantiate(cfg.nn.data.datasets.train, split="train", _recursive_=False)


if __name__ == "__main__":
    main()
