import pytorch_lightning as pl
from data import Data
from models import HuggingFaceModel, Model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM


class PreTrainedBERTModel(Model, HuggingFaceModel):
    def __init__(self, dataset: Data) -> "Model":
        self.pretrained_model_name = "bert-base-uncased"
        model = AutoModelForMaskedLM.from_pretrained(self.pretrained_model_name)
        optimizer = AdamW(model.parameters(), lr=5e-5)

        super().__init__(model, optimizer)

        self.dataset = dataset

    def fit(self):
        trainer = pl.Trainer(max_epochs=10)
        train_dataloader = self.preprocess_data(dataset=self.dataset, shuffle=True)
        trainer.fit(self, train_dataloaders=train_dataloader)
