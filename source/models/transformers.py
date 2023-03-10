from model import Model
from transformers import (
    AutoModelForMaskedLM,
    PreTrainedTokenizer,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data import Data


class PreTrainedBERTModel(Model):
    def __init__(self) -> "Model":
        self.pretrained_model_name = "bert-base-uncased"
        model = AutoModelForMaskedLM.from_pretrained(self.pretrained_model_name)
        optimizer = AdamW(model.parameters(), lr=5e-5)

        super().__init__(model, optimizer)
