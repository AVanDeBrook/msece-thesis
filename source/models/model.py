import os
import random
from typing import *

import pytorch_lightning as pl
import torchmetrics
import torch
import numpy
from data import Data
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


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
        self.test_perplexity = torchmetrics.Perplexity()

        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []

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

    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)

        self.test_perplexity.update(outputs.logits, batch["labels"])
        self.test_loss.append(outputs.loss)

        return outputs.loss

    def on_test_epoch_end(self):
        test_mean = torch.stack(self.test_loss).mean()
        self.log("test_ppl", self.test_perplexity)
        self.log("test_loss", test_mean)

        self.test_loss.clear()

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

    def fit(self, max_epochs: int = 1):
        raise NotImplementedError()


class HuggingFaceModel:
    def __init__(self):
        pass

    def preprocess_data(self, dataset: Data):
        # function to initialize DataLoader workers and control randomness
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)

        assert self.tokenizer is not None
        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm_probability=0.2)

        # from: https://pytorch.org/docs/stable/notes/randomness#dataloader
        g = torch.Generator()
        g.manual_seed(1)

        dataset.preprocess(tokenizer=self.tokenizer, collator=collator)

        return DataLoader(
            dataset,
            batch_size=16,
            drop_last=True,
            num_workers=os.cpu_count(),
            worker_init_fn=seed_worker,
            generator=g,
        )
