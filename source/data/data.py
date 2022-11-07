import os
from typing import *
from pathlib import Path


class Data(object):
    _required_fields: List[str]

    def __init__(self, data_root: str):
        assert isinstance(data_root, str)
        assert os.path.exists(data_root)

    def parse_transcripts(self) -> Dict[str, Union[str, float]]:
        raise NotImplementedError()
