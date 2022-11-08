import os
import seaborn.objects as so
from typing import *


class Data(object):
    """
    Attributes:
    -----------

    """
    _required_fields: List[str]
    _manifest_data: List[Dict[str, Union[float, str]]]

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

    def create_utterance_hist(self, utterance_counts=[]):
        """

        """
        # check if manifest data has been generated, parse transcripts and generate
        # manifest data if not
        if len(self._manifest_data) == 0:
            self.parse_transcripts()

        # check if utterance counts (optional arg) has been provided, calculate utterance
        # counts from transcriptions
        if len(utterance_counts) == 0:
            for data in self._manifest_data:
                words = data["text"].split(" ")
                utterance_counts.append(len(words))

        p = so.Plot(utterance_counts).add(so.Bar(), so.Hist())
        return p.plot()


    def create_spectrograms(self):
        pass

    def calc_utterance_stats(self):
        pass
