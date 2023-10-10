import pytorch_lightning as pl
from data import Data, PLDataLoader
from models import HuggingFaceModel, Model
from torch import nn
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer


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

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)

        optimizer = AdamW(model.parameters(), lr=4e-5, betas=[0.9, 0.98], eps=1e-6, weight_decay=0.01)

        super().__init__(
            model=model, optimizer=optimizer, checkpoint_name="bert_finetuned"
        )

        if train_dataset is not None:
            self.train_dataset = train_dataset
        if valid_dataset is not None:
            self.valid_dataset = valid_dataset

    def fit(self, max_epochs: int = 1):
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu")
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)

        optimizer = AdamW(model.parameters(), lr=4e-5, betas=[0.9, 0.98], eps=1e-6, weight_decay=0.01)

        super().__init__(
            model=model, optimizer=optimizer, checkpoint_name="roberta_finetuned"
        )

        if train_dataset is not None:
            self.train_dataset = train_dataset
        if valid_dataset is not None:
            self.valid_dataset = valid_dataset

    def fit(self, max_epochs: int = 1):
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu")
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
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.config = AutoConfig.from_pretrained(
                "bert-base-uncased", vocab_size=len(self.tokenizer)
            )
            model = AutoModelForMaskedLM.from_config(self.config)

        optimizer = AdamW(model.parameters(), lr=4e-5, betas=[0.9, 0.98], eps=1e-6, weight_decay=0.01)

        super().__init__(
            model=model, optimizer=optimizer, checkpoint_name="bert_randominit"
        )

        if train_dataset is not None:
            self.train_dataset = train_dataset

            self.tokenizer.train_new_from_iterator(
                self.train_dataset.data, vocab_size=self.config.vocab_size
            )

        if valid_dataset is not None:
            self.valid_dataset = valid_dataset

    def fit(self, max_epochs: int = 1):
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu")
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
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self.config = AutoConfig.from_pretrained("roberta-base")
            model = AutoModelForMaskedLM.from_config(self.config)

        optimizer = AdamW(model.parameters(), lr=4e-5, betas=[0.9, 0.98], eps=1e-6, weight_decay=0.01)

        super().__init__(
            model=model, optimizer=optimizer, checkpoint_name="roberta_randominit"
        )

        if train_dataset is not None:
            self.train_dataset = train_dataset

            self.tokenizer.train_new_from_iterator(
                self.train_dataset.data, vocab_size=self.config.vocab_size
            )

        if valid_dataset is not None:
            self.valid_dataset = valid_dataset

    def fit(self, max_epochs: int = 1):
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu")
        datamodule = PLDataLoader(
            train_dataset=self.train_dataset,
            val_dataset=self.valid_dataset,
            preprocess_fn=self.preprocess_data,
        )
        trainer.fit(self, datamodule=datamodule)
