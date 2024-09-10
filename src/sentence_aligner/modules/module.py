from typing import Any, Dict

import pytorch_lightning as pl
import torch
from sentence_transformers import SentenceTransformer

class TextEncoder(pl.LightningModule):
    def __init__(self, transformer_name: str, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(
            logger=False, ignore=("metadata",)
        )  # populate self.hparams with args and kwargs automagically!
        self.transformer: SentenceTransformer = self.hparams.transformer

        if self.hparams.freeze_encoder:
            for module in self.transformer.modules():
                for param in module.parameters():
                    param.requires_grad = False
        self.dropout = torch.nn.Dropout(self.hparams.dropout)

    def forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        """Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """

        inputs = dict()

        inputs["input_ids"], inputs["attention_mask"] = (
            batch["input_ids"],
            batch["attention_mask"],
        )
        transformer_out = self.transformer(inputs)

        return dict(encoding=transformer_out["sentence_embedding"])
