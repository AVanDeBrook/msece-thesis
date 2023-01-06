import os
import glob
import subprocess
from typing import *
from data import Data


class ZCUATCDataset(Data):
    """ """

    def __init__(self, data_root: str, **kwargs):
        super(ZCUATCDataset, self).__init__(data_root, **kwargs)

    def parse_transcripts(self) -> Dict[str, Union[str, float]]:
        pass

    @property
    def name(self) -> str:
        return "ZCU ATC"
