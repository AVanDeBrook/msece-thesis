import os

import pytorch_lightning as pl
import torch
from data import Data, PLDataLoader
from models import HuggingFaceModel, Model
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM


class PreTrainedBERTModel(Model, HuggingFaceModel):
    def __init__(
        self,
        train_dataset: Data = None,
        valid_dataset: Data = None,
        model: nn.Module = None,
    ) -> "Model":
        self.pretrained_model_name = "bert-base-uncased"

        if model is None:
            model = AutoModelForMaskedLM.from_pretrained(self.pretrained_model_name)

        optimizer = AdamW(model.parameters(), lr=5e-5)

        super().__init__(model, optimizer)

        if train_dataset is not None:
            self.train_dataset = train_dataset
        if valid_dataset is not None:
            self.valid_dataset = valid_dataset

    def fit(self):
        trainer = pl.Trainer(max_epochs=1, accelerator="gpu")
        datamodule = PLDataLoader(
            train_dataset=self.train_dataset,
            val_dataset=self.valid_dataset,
            preprocess_fn=self.preprocess_data,
        )
        trainer.fit(self, datamodule=datamodule)

    def save_to(self, path: str):
        if self.model is None:
            raise ValueError("model has not been initialized, nothing to save")

        os.makedirs(path, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(path, f"{str(self.__class__.__name__)}.pt"),
        )

    @classmethod
    def load_from(
        cls, path: str, train_dataset: Data = None, valid_dataset: Data = None
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find path specified: '{path}'")

        model = cls(train_dataset=train_dataset, valid_dataset=valid_dataset)

        model.model.load_state_dict(
            torch.load(os.path.join(path, f"{str(model.__class__.__name__)}.pt"))
        )

        return model
