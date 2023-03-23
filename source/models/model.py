import os
from typing import *

import pytorch_lightning as pl
import torchmetrics
import torch
from data import Data
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


class Model(pl.LightningModule):
    def __init__(
        self, model: nn.Module, optimizer: optim.AdamW, checkpoint_name: str = None
    ) -> "Model":
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer)
        self.checkpoint_name = checkpoint_name

        self.train_perplexity = torchmetrics.Perplexity()
        self.valid_perplexity = torchmetrics.Perplexity()

        self.train_loss = []
        self.valid_loss = []

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)

        self.train_perplexity.update(outputs.logits, batch["labels"])
        self.train_loss.append(outputs.loss)

        return outputs.loss

    def on_train_epoch_end(self):
        epoch_mean = torch.stack(self.train_loss).mean()
        self.log("train_ppl", self.train_perplexity)
        self.log("train_loss", epoch_mean)
        self.log("learning_rate", self.scheduler.get_last_lr()[0])

        self.train_loss.clear()

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)

        self.valid_perplexity.update(outputs.logits, batch["labels"])
        self.valid_loss.append(outputs.loss)

        return outputs.loss

    def on_validation_epoch_end(self):
        epoch_mean = torch.stack(self.valid_loss).mean()
        self.log("val_ppl", self.valid_perplexity)
        self.log("val_loss", epoch_mean)

        self.valid_loss.clear()

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_loss",
            },
        }

    def save_to(self, path: str):
        if self.model is None:
            raise ValueError("model has not been initialized, nothing to save")

        os.makedirs(path, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(path, f"{self.checkpoint_name}.pt"),
        )

    @classmethod
    def load_from(cls, path: str, **kwargs):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find path specified: '{path}'")

        model = cls(**kwargs)

        model.model.load_state_dict(
            torch.load(os.path.join(path, f"{model.checkpoint_name}.pt"))
        )

        return model

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
            dataset,
            batch_size=16,
            drop_last=True,
            shuffle=shuffle,
            num_workers=os.cpu_count(),
        )
