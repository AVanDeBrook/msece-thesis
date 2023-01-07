import os
import matplotlib
import matplotlib.collections
import matplotlib.figure
import seaborn.objects
import seaborn.objects as so
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from typing import *
from pathlib import Path


TokenStats = namedtuple("TokenStats", ["tokens", "samples"])
"""
Named tuple for ease of access and return for utterance statistics for the dataset.

`tokens` corresponds to the total number of tokens found in the dataset

`samples` corresponds to the number of samples in the dataset
"""


class Data:
    """
    Top-level Data class that provides several methods for processing and analyzing text datasets for NLP processes.

    This class should be extended and the following methods/properties implemented for each dataset:
    * `parse_transcripts`
    * `name`

    Attributes:
    -----------
    `_manifest_data`: list of dictionary objects. Each object corresponds to one data sample
    and typically contains the following metadata:
    * `audio_filepath` (required) - path to the audio data (input data). Type: `str`,
    absolute file path, conforms to `os.PathLike`
    * `duration` (required) - duration, in seconds, of the audio data. Type: `float`
    * `text` (required) - transcript of the audio data (label/ground truth). Type: `str`
    * `offset` - if more than one sample is present in a single audio file, this field
    specifies its offset i.e. start time in the audio file. Type: `float`

    `_random`: numpy seeded RNG instance

    `_normalized`: bool indicating whether samples in the dataset have been normalized/preprocessed
    """

    data: List[Dict[str, Union[float, str]]]
    _random: np.random.Generator
    _normalized: bool

    def __init__(self, data_root: str, random_seed: int = None):
        """
        Arguments:
        ----------
        `data_root`: path to the base of the dataset, basically just a path from which the
        audio and transcript data can be found. Varies by dataset and implementation.
        """
        assert isinstance(data_root, str)
        assert os.path.exists(data_root)

        # create random number generator sequence with specified seed, if applicable
        Data._random = np.random.default_rng(random_seed)

    def parse_transcripts(self) -> List[str]:
        """
        This method must be overridden and implemented for each implementation of this class
        for datasets.

        Returns:
        --------
        Dictionary (from `json` module) with necessary data info e.g. annotations, file
        path, audio length, offset.
        """
        raise NotImplementedError(
            "This is an abstract method that should be implemented and overridden for "
            "all classes that implement this one. If this method has been called, there "
            "is an issue with the class that extended the Data class."
        )

    def create_token_hist(
        self,
        utterance_counts: List[int] = [],
        plot_type: Literal["matplotlib", "seaborn"] = "seaborn",
    ) -> Union[seaborn.objects.Plot, matplotlib.figure.Figure]:
        """
        Calculates the number of utterances in each sample and generates a histogram.

        Utterance counts are determined by splitting each transcript on whitespace and
        calculating the length of the resulting list.

        TODO: add flexibility for plot types.

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
        if len(self.data) == 0:
            self.parse_transcripts()

        # check if utterance counts (optional arg) has been provided, calculate utterance
        # counts from transcriptions
        if len(utterance_counts) == 0:
            for data in self.data:
                words = data["text"].split(" ")
                utterance_counts.append(len(words))

        p = so.Plot(utterance_counts).add(so.Bar(), so.Hist())
        p.label()
        return p

    def calc_token_stats(self) -> TokenStats:
        """
        Calculate the following:
        * Total number of utterances in the data
        * Total number of samples in the data
        * Cumulative duration of samples

        Returns:
        --------
        an `TokenStats` named tuple with `tokens` and `samples` field corresponding to total utterance counts, total sample duration,
        and total samples in the data
        """
        # check if manifest data has been generated
        if len(self.data) == 0:
            self.parse_transcripts()

        total_token_count = 0

        for data in self.data:
            utterances = data["text"].split(" ")
            total_token_count += len(utterances)

        return TokenStats(
            utterances=total_token_count,
            samples=len(self.data),
        )

    def token_freq_analysis(self, normalize=False) -> Dict[str, Union[int, float]]:
        """
        Perform a token frequency analysis on the dataset (number of occurrences of each token throughout the dataset).

        Arguments:
        ----------
        `normalize`: (optional)`bool`, whether to normalize values such that all frequencies add to 1.

        Returns:
        --------
        `token_freqs` `dict` with tokens and number of occurrences of those tokens throughout the dataset.
        """
        if len(self.data) == 0:
            self.parse_transcripts()

        token_freqs = {}

        for sample in self.data:
            sample = sample["text"]
            for token in sample.split():
                if token in token_freqs.keys():
                    token_freqs[token] += 1
                else:
                    token_freqs[token] = 1

        if normalize:
            num_tokens = len(token_freqs)
            for token, freq in token_freqs.items():
                token_freqs[token] = float(freq) / num_tokens

        return token_freqs

    def normalize_data(self):
        pass

    def dump_corpus(self, outfile: str, make_dirs: bool = True):
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
        outfile = Path(outfile).absolute()
        os.makedirs(str(outfile.parent), exist_ok=make_dirs)

        # check if manifest data has been generated
        if len(self.data) == 0:
            self.parse_transcripts()

        # write each data point its own line in the file, in json format (conform to NeMo
        # manifest specification)
        with open(str(outfile), "w") as manifest:
            for entry in self.data:
                manifest.write(entry)
                manifest.write("\n")

    @property
    def name(self) -> str:
        raise NotImplementedError(
            "This property should be implemented by the extending class"
        )
