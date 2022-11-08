import os
from typing import *


class Data(object):
    """
    Attributes:
    -----------

    """
    _required_fields: List[str]

    def __init__(self, data_root: str):
        """
        Arguments:
        ----------
        `data_root`: path to the base of the dataset, basically just a path from which the audio and transcript data can be found.
                     Varies by dataset and implementation.
        """
        assert isinstance(data_root, str)
        assert os.path.exists(data_root)

    def parse_transcripts(self) -> Dict[str, Union[str, float]]:
        """
        This method must be overridden and implemented for each implementation of this class
        for datasets.

        Returns:
        --------
        Dictionary (from `json` module) with necessary data info e.g. annotations, file path, audio length, offset
        """
        raise NotImplementedError()

    def create_utterance_hist(self):
        pass

    def create_spectrograms(self):
        pass

    def calc_utterance_stats(self):
        pass
