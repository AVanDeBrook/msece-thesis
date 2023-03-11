from typing import *

import pytorch_lightning as pl
from data import Data
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


class Model(pl.LightningModule):
    def __init__(self, model: nn.Module, optimizer: optim.AdamW) -> "Model":
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        return outputs.loss

    def configure_optimizers(self):
        return self.optimizer

    def fit(self):
        raise NotImplementedError()


class HuggingFaceModel:
    def __init__(self):
        pass

    def preprocess_data(self, dataset: Data, shuffle: bool = True):
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.2)

        dataset.preprocess(tokenizer=tokenizer, collator=collator)

        return DataLoader(dataset, batch_size=8)
