import pytorch_lightning as pl
from data import Data
from models import HuggingFaceModel, Model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM


class PreTrainedBERTModel(Model, HuggingFaceModel):
    def __init__(self, train_dataset: Data, valid_dataset: Data) -> "Model":
        self.pretrained_model_name = "bert-base-uncased"
        model = AutoModelForMaskedLM.from_pretrained(self.pretrained_model_name)
        optimizer = AdamW(model.parameters(), lr=5e-5)

        super().__init__(model, optimizer)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def fit(self):
        trainer = pl.Trainer(max_epochs=10, accelerator="gpu")
        train_dataloader = self.preprocess_data(
            dataset=self.train_dataset, shuffle=True
        )
        validation_dataloader = self.preprocess_data(
            dataset=self.valid_dataset, shuffle=True
        )
        trainer.fit(
            self,
            train_dataloaders=train_dataloader,
            val_dataloaders=validation_dataloader,
        )
