from typing import Any, Dict, Optional, Sequence, Tuple, Union

import hydra
import hydra._internal.instantiate._instantiate2
import hydra.types
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.optim import Optimizer
from torchmetrics import Accuracy, F1Score, Precision, Recall
from transformers import AutoConfig, DistilBertConfig, DistilBertModel

from sentence_aligner.data.datamodule import MetaData
from sentence_aligner.modules.module import TextEncoder

instantiate = hydra._internal.instantiate._instantiate2.instantiate
call = instantiate


class MyLightningModule(pl.LightningModule):
    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(
            logger=False, ignore=("metadata",)
        )
        self.metadata: MetaData = metadata
        self.config = AutoConfig.from_pretrained(self.hparams.transformer_name)

        self.precomputed_embeddings: bool = self.hparams.precomputed_embeddings

        self.transformer: SentenceTransformer = SentenceTransformer(
            self.hparams.transformer_name
        )

        self.sentence_encoder = TextEncoder(
            transformer_name=self.hparams.transformer_name,
            transformer=self.transformer,
            freeze_encoder=self.hparams.freeze_encoder,
            dropout=0.0,
        )
            
        self.hidden_size = self.config.hidden_size

        self.transfomer_sentence_context_config = DistilBertConfig(
            n_heads=self.hparams.sentence_context_n_heads,
            n_layers=self.hparams.sentence_context_n_layers,
            dropout=self.hparams.dropout,
        )
        self.transfomer_sentence_context = DistilBertModel(
            self.transfomer_sentence_context_config
        )

        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size // 2),
            torch.nn.Dropout(self.hparams.dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size // 2, 1),
        )

        # self.activation = instantiate(self.hparams.activation)
        self.val_metrics: torch.nn.ModuleDict = torch.nn.ModuleDict(
            {
                "acc": Accuracy(task="binary"),
                "recall": Recall(task="binary"),
                "prec": Precision(task="binary"),
                "f1": F1Score(task="binary"),
            }
        )

        self.test_metrics: torch.nn.ModuleDict = torch.nn.ModuleDict(
            {
                "acc": Accuracy(task="binary"),
                "recall": Recall(task="binary"),
                "prec": Precision(task="binary"),
                "f1": F1Score(task="binary"),
            }
        )

    def forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """

        inputs = dict()

        inputs["input_ids"], inputs["attention_mask"] = (
            batch["merged_enc"]["input_ids"],
            batch["merged_enc"]["attention_mask"],
        )
        transformer_out = self.transformer(**inputs)
        cls_enc = transformer_out["last_hidden_state"][:, 0, :]
        return dict(cls_enc=cls_enc)

    def step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        compute_loss: bool = True,
        compute_metrics: bool = False,
    ):
        # Use pre-computed sentence embeddings or compute them at runtime
        if self.precomputed_embeddings:
            encoding_sources = batch["sources_embeds"]
            encoding_targets = batch["targets_embeds"]
        else:
            model_out_sources = self.sentence_encoder(batch["sources_encodings"])
            model_out_targets = self.sentence_encoder(batch["targets_encodings"])
            encoding_sources = model_out_sources["encoding"]
            encoding_targets = model_out_targets["encoding"]

        # output transformer
        encoding_sources = self.transfomer_sentence_context(
            inputs_embeds=encoding_sources.unsqueeze(dim=0)
        )["last_hidden_state"].squeeze(dim=0)
        encoding_targets = self.transfomer_sentence_context(
            inputs_embeds=encoding_targets.unsqueeze(dim=0)
        )["last_hidden_state"].squeeze(dim=0)

        cartesian_matrix_index = torch.cartesian_prod(
            torch.arange(0, len(encoding_sources), device=encoding_sources.device),
            torch.arange(0, len(encoding_targets), device=encoding_sources.device),
        )
        x_, y_ = cartesian_matrix_index[:, 0], cartesian_matrix_index[:, 1]
        cartesian_matrix_prod = encoding_sources[x_] * encoding_targets[y_]
        classification_output = self.classification_head(cartesian_matrix_prod)
        result = dict()

        if compute_loss and "gold_matrix" in batch:
            result["loss"] = F.binary_cross_entropy_with_logits(
                classification_output, batch["gold_matrix"].float().unsqueeze(-1)
            )
        result["predictions"] = F.sigmoid(classification_output)
        result["matrix_index"] = cartesian_matrix_index

        return result

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        all_losses = []
        for batch_s in batch:
            step_out = self.step(
                batch_s, batch_idx, split="train", compute_metrics=False
            )
            if step_out is not None:
                loss = step_out["loss"]
                all_losses.append(loss)
        if len(all_losses) > 0:
            loss = torch.stack(all_losses).mean()
            self.log_dict(
                {"train/loss": loss},
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            return loss
        else:
            return None

    def validation_step(
        self, batch: Any, batch_idx: int, dataset_index: int = -1
    ) -> torch.Tensor:
        step_out = self.step(batch, batch_idx, split=f"val", compute_metrics=True)
        if step_out is not None:
            loss = step_out["loss"]

            return dict(
                loss=loss,
                predictions=step_out["predictions"],
                matrix_index=step_out["matrix_index"],
                gold_matrix=batch["gold_matrix"],
                sources_text=batch["sources_text"],
                targets_text=batch["targets_text"],
                sources_ids=batch["sources_ids"],
                targets_ids=batch["targets_ids"],
            )
        return None

    def validation_epoch_end(self, outputs):
        # task_name = self.metadata.dataset_names["val"][0].replace(".jsonl", "").replace("val_", "")
        columns = ["s_ids", "t_ids", "source", "targets"]
        all_losses = []
        f1_list = []

        for output, ds_name in zip(outputs, self.metadata.dataset_names["val"]):
            task_name = ds_name.replace("val_", "").replace(".jsonl", "")
            all_preds = []
            all_golds = []
            data = []
            if isinstance(output, list):  ## More than one file in the validation.
                for out in output:
                    all_losses.append(out["loss"])
                    preds = torch.round(out["predictions"].flatten())
                    prediciton_indices = out["matrix_index"][preds.bool()]
                    sources_text = out["sources_text"]
                    targets_text = out["targets_text"]
                    sources_ids = out["sources_ids"]
                    targets_ids = out["targets_ids"]
                    prediction_couples = dict()
                    for s_idx, t_idx in prediciton_indices.detach().tolist():
                        prediction_couples.setdefault(s_idx, []).append(t_idx)
                    for s_idx, targets in prediction_couples.items():
                        s_text = sources_text[s_idx]
                        t_texts = [targets_text[t] for t in targets]
                        t_ids = [targets_ids[t] for t in targets]
                        data.append(
                            [
                                sources_ids[s_idx],
                                " -- ".join(t_ids),
                                s_text,
                                " -- ".join(t_texts),
                            ]
                        )
                        # print(s_text, targets)
                    all_preds.append(preds)
                    all_golds.append(out["gold_matrix"])
            else:  ## Only 1 file in the validaton.
                all_losses.append(output["loss"])
                preds = torch.round(output["predictions"].flatten())
                prediciton_indices = output["matrix_index"][preds.bool()]
                sources_text = output["sources_text"]
                targets_text = output["targets_text"]
                sources_ids = output["sources_ids"]
                targets_ids = output["targets_ids"]
                prediction_couples = dict()
                for s_idx, t_idx in prediciton_indices.detach().tolist():
                    prediction_couples.setdefault(s_idx, []).append(t_idx)
                for s_idx, targets in prediction_couples.items():
                    s_text = sources_text[s_idx]
                    t_texts = [targets_text[t] for t in targets]
                    t_ids = [targets_ids[t] for t in targets]
                    data.append(
                        [
                            sources_ids[s_idx],
                            " -- ".join(t_ids),
                            s_text,
                            " -- ".join(t_texts),
                        ]
                    )
                    # print(s_text, targets)
                all_preds.append(preds)
                all_golds.append(output["gold_matrix"])

            # self.logger.log_table(columns=columns, data=data,
            #                     key=f"alignment_table")
            loss = torch.stack(all_losses).mean()
            self.log_dict(
                {"val/loss": loss},
                on_epoch=True,
                prog_bar=True,
            )
            all_preds = torch.cat(all_preds, dim=0)
            all_golds = torch.cat(all_golds, dim=0)
            # acc = self.val_metrics["acc"](all_preds, all_golds)
            f1 = self.val_metrics["f1"](all_preds, all_golds)
            recall = self.val_metrics["recall"](all_preds, all_golds)
            prec = self.val_metrics["prec"](all_preds, all_golds)
            # self.log(f"val_acc_{task_name}", acc)
            self.log(f"val_recall_{task_name}", recall)
            self.log(f"val_prec_{task_name}", prec)
            self.log(f"val_f1_{task_name}", f1)

            f1_list.append(f1)

        val_f1_full = torch.stack(f1_list).mean()
        self.log(f"val_f1_full", val_f1_full)

        return torch.stack(all_losses).mean()

    def test_epoch_end(self, outputs):
        all_losses = []
        f1_list = []

        for output, ds_name in zip(outputs, self.metadata.dataset_names["test"]):
            task_name = ds_name.replace("test_", "").replace(".jsonl", "")
            all_preds = []
            all_golds = []
            if isinstance(output, list):  ## Same as validation.
                for out in output:
                    all_losses.append(out["loss"])
                    preds = torch.round(out["predictions"].flatten())
                    all_preds.append(preds)
                    all_golds.append(out["gold_matrix"])
            else:
                all_losses.append(output["loss"])
                preds = torch.round(output["predictions"].flatten())
                all_preds.append(preds)
                all_golds.append(output["gold_matrix"])
            all_preds = torch.cat(all_preds, dim=0)
            all_golds = torch.cat(all_golds, dim=0)
            # acc = self.test_metrics["acc"](all_preds, all_golds)
            f1 = self.test_metrics["f1"](all_preds, all_golds)
            recall = self.test_metrics["recall"](all_preds, all_golds)
            prec = self.test_metrics["prec"](all_preds, all_golds)
            # self.log(f"test_acc_{task_name}", acc)
            self.log(f"test_recall_{task_name}", recall)
            self.log(f"test_prec_{task_name}", prec)
            self.log(f"test_f1_{task_name}", f1)

            f1_list.append(f1)

        test_f1_full = torch.stack(f1_list).mean()
        self.log(f"test_f1_full", test_f1_full)

        return torch.stack(all_losses).mean()

    def test_step(
        self, batch: Any, batch_idx: int, dataset_index: int = None
    ) -> torch.Tensor:
        step_out = self.step(
            batch, batch_idx, split="test", compute_metrics=True, compute_loss=False
        )
        loss = torch.zeros(1)

        return dict(
            loss=loss,
            predictions=step_out["predictions"],
            matrix_index=step_out["matrix_index"],
            gold_matrix=batch["gold_matrix"],
        )

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters(), _convert_="partial"
        )
        self.hparams.lr_scheduler.num_warmup_steps = (
            self.hparams.warmup_percentage
            * self.hparams.lr_scheduler.num_training_steps
        )
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
