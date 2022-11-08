import glob
from typing import *
from pathlib import Path
from data import Data
from data import hiwireutils


class HIWIREData(Data):

    def __init__(self, data_root: str):
        super(Data, self).__init__(data_root)

    def parse_transcripts(self) -> Dict[str, Union[str, float]]:
        pass
