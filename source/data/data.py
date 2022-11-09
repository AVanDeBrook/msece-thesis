import os
import json
import librosa
import seaborn.objects as so
import matplotlib.pyplot as plt
from collections import namedtuple
from typing import *


UtteranceStats = namedtuple(
    "UtteranceStats", ["utterances", "sample_duration", "samples"]
)
"""
Named tuple for ease of access and return for utterance statistics for the dataset.

`utterances` corresponds to the total number of utterances found in the dataset

`sample_duration` corresponds to the cumulative duration of the samples in the dataset

`samples` corresponds to the number of samples in the dataset
"""


class Data(object):
    """
    Attributes:
    -----------

    Methods:
    --------
    """

    _required_fields: List[str]
    _manifest_data: List[Dict[str, Union[float, str]]]

    def __init__(self, data_root: str):
        """
        Arguments:
        ----------
        `data_root`: path to the base of the dataset, basically just a path from which the
        audio and transcript data can be found. Varies by dataset and implementation.
        """
        assert isinstance(data_root, str)
        assert os.path.exists(data_root)

    def parse_transcripts(self) -> Dict[str, Union[str, float]]:
        """
        This method must be overridden and implemented for each implementation of this class
        for datasets.

        Returns:
        --------
        Dictionary (from `json` module) with necessary data info e.g. annotations, file
        path, audio length, offset.
        """
        raise NotImplementedError()

    def create_utterance_hist(
        self,
        utterance_counts: List[int] = [],
        plot_type: Literal["matplotlib", "seaborn"] = "seaborn",
    ) -> Union[so.Plot, plt.Figure]:
        """
        Calculates the number of utterances in each sample and generates a histogram.

        Utterance counts are determined by splitting each transcript on whitespace and
        calculating the length of the resulting list.

        Arguments:
        ----------
        `utterance_counts`: (Optional) `list` of `ints` for precalculated utterance counts.

        `plot_type`: (Optional) `str` type of plot tool to use to create the histogram.
        Can be either `"seaborn"` or `"matplotlib"`. Defaults to `"seaborn"`.

        Returns:
        --------
        Either a `matplotlib.pyplot.Figure` or `seaborn.object.Plot` instance, depending on the value of `plot_type`.
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
        """

        """
        pass

    def calc_utterance_stats(self) -> UtteranceStats:
        """
        Calculate the following:
        * Total number of utterances in the data
        * Total number of samples in the data
        * Cumulative duration of samples

        Returns:
        --------
        an `UtteranceStats` named tuple with `utterances`, `sample_duration`, and
        `samples` field corresponding to total utterance counts, total sample duration,
        and total samples in the data
        """
        # check if manifest data has been generated
        if len(self._manifest_data) == 0:
            self.parse_transcripts()

        total_utterance_count = 0
        total_sample_duration = 0.0

        for data in self._manifest_data:
            # utterances are essentially just words in the transcripts, splitting on
            # whitespace accomplishes the goal
            utterances = data["text"].split(" ")
            duration = data["duration"]

            total_utterance_count += len(utterances)
            total_sample_duration += duration

        return UtteranceStats(
            utterances=total_utterance_count,
            sample_duration=total_sample_duration,
            samples=len(self._manifest_data),
        )

    def dump_manifest(self, outfile: str, make_dirs: bool = True):
        """
        Dump input data paths, labels, and metadata to `outfile` in NeMo manifest format.

        Arguments:
        ----------
        `outfile`: `str`, output path

        `make_dirs`: (optional) `bool`, whether to make nonexistent parent directories
        in `outfile`. Defaults to `True`.

        Returns:
        --------
        None
        """
        os.makedirs(outfile, exist_ok=make_dirs)

        # check if manifest data has been generated
        if len(self._manifest_data) == 0:
            self.parse_transcripts()

        # write each data point its own line in the file, in json format (conform to NeMo
        # manifest specification)
        with open(outfile, "w") as manifest:
            for entry in self._manifest_data:
                manifest.write(json.dumps(entry))
                manifest.write("\n")
