import os
from typing import *

import pytorch_lightning as pl
import torchmetrics
from data import Data
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorForLanguageModeling


class Model(pl.LightningModule):
    def __init__(self, model: nn.Module, optimizer: optim.AdamW) -> "Model":
        super().__init__()

        self.model = model
        self.optimizer = optimizer

        self.train_perplexity = torchmetrics.Perplexity()
        self.valid_perplexity = torchmetrics.Perplexity()

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)

        step_perplexity = self.train_perplexity(outputs.logits, batch["labels"])
        self.log("train_step_ppl", step_perplexity)

        return outputs.loss

    # def on_train_epoch_end(self):
    #     self.log("train_epoch_ppl", self.train_perplexity)

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.valid_perplexity.update(outputs.logits, batch["labels"])
        return outputs.loss

    # def on_validation_epoch_end(self):
    #     self.log("val_ppl", self.valid_perplexity)

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.LinearLR(self.optimizer),
                "monitor": "val_loss",
            },
        }

    def fit(self):
        raise NotImplementedError()


class HuggingFaceModel:
    def __init__(self):
        pass

    def preprocess_data(self, dataset: Data, shuffle: bool = True):
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.2)

        dataset.preprocess(tokenizer=tokenizer, collator=collator)

        return DataLoader(
            dataset, batch_size=16, drop_last=True, num_workers=os.cpu_count()
        )