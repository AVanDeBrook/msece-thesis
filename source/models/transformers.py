import pytorch_lightning as pl
from data import Data, PLDataLoader
from models import HuggingFaceModel, Model
from torch import nn
from torch.optim import AdamW
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    RobertaConfig,
    RobertaForMaskedLM,
)


class PreTrainedBERTModel(Model, HuggingFaceModel):
    def __init__(
        self,
        train_dataset: Data = None,
        valid_dataset: Data = None,
        model: nn.Module = None,
        **kwargs
    ) -> "Model":
        self.pretrained_model_name = "bert-base-uncased"

        if model is None:
            model = AutoModelForMaskedLM.from_pretrained(self.pretrained_model_name)

        optimizer = AdamW(model.parameters(), lr=4e-5)

        super().__init__(
            model=model, optimizer=optimizer, checkpoint_name="bert_finetuned"
        )

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


class PreTrainedRoBERTaModel(Model, HuggingFaceModel):
    def __init__(
        self,
        train_dataset: Data = None,
        valid_dataset: Data = None,
        model: nn.Module = None,
        **kwargs
    ) -> "Model":
        self.pretrained_model_name = "roberta-base"

        if model is None:
            model = AutoModelForMaskedLM.from_pretrained(self.pretrained_model_name)

        optimizer = AdamW(model.parameters(), lr=4e-5)

        super().__init__(
            model=model, optimizer=optimizer, checkpoint_name="roberta_finetuned"
        )

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


class RandomInitBERTModel(Model, HuggingFaceModel):
    def __init__(
        self,
        train_dataset: Data = None,
        valid_dataset: Data = None,
        model: nn.Module = None,
        **kwargs
    ) -> "Model":
        self.pretrained_model_name = "bert-base-uncased"
        if model is None:
            model = BertForMaskedLM(BertConfig())

        optimizer = AdamW(model.parameters(), lr=4e-5)

        super().__init__(
            model=model, optimizer=optimizer, checkpoint_name="bert_randominit"
        )

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


class RandomInitRoBERTaModel(Model, HuggingFaceModel):
    def __init__(
        self,
        train_dataset: Data = None,
        valid_dataset: Data = None,
        model: nn.Module = None,
        **kwargs
    ) -> "Model":
        self.pretrained_model_name = "roberta-base"

        if model is None:
            model = RobertaForMaskedLM(RobertaConfig(vocab_size=50265))

        optimizer = AdamW(model.parameters(), lr=4e-5)

        super().__init__(
            model=model, optimizer=optimizer, checkpoint_name="roberta_randominit"
        )

        if train_dataset is not None:
            self.train_dataset = train_dataset
        if valid_dataset is not None:
            self.valid_dataset = valid_dataset

    def fit(self):
        trainer = pl.Trainer(max_epochs=1, accelerator="cpu")
        datamodule = PLDataLoader(
            train_dataset=self.train_dataset,
            val_dataset=self.valid_dataset,
            preprocess_fn=self.preprocess_data,
        )
        trainer.fit(self, datamodule=datamodule)
