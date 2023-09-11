import os
from pathlib import Path
from typing import *
from copy import deepcopy

import matplotlib
import matplotlib.collections
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from torch.utils.data import IterableDataset
from transformers import (
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling,
)
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class Data(IterableDataset):
    """
    Top-level Data class that provides several methods for processing and analyzing text datasets for NLP processes.

    This class should be extended and the following methods/properties implemented for each dataset:
    * `parse_transcripts`

    Attributes:
    -----------
    `_random`: numpy seeded RNG instance
    """

    _random: np.random.Generator

    def __init__(self, random_seed: int = 1, dataset_name: str = "data"):
        """
        Arguments:
        ----------
        `data_root`: path to the base of the dataset, basically just a path from which the
        audio and transcript data can be found. Varies by dataset and implementation.
        """
        self.dataset_name: str = dataset_name
        self.data: List[str] = []
        # create random number generator sequence with specified seed, if applicable
        Data._random: np.random.Generator = np.random.default_rng(random_seed)

    def __iter__(self):
        assert len(self.data) != 0
        assert self.tokenized_data is not None

        return iter(self.tokenized_data)

    def __len__(self):
        return len(self.tokenized_data)

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
            "is an issue with the implementing class."
        )

    def preprocess(
        self, tokenizer: PreTrainedTokenizer, collator: DataCollatorForLanguageModeling
    ) -> List[List[int]]:
        """
        Preprocesses the data using the given tokenizer (`tokenizer`). Pads the sequence to a
        maximum length (if the sequence is below the required length) or truncates
        (`padding="max_length"` and `truncation=True` parameters).

        Arguments:
        ----------
        `tokenizer`: a ~`transformers.PreTrainedTokenizer` that will be used to tokenize the data
        in the dataset.

        Return:
        -------
        list of tokenized data (list of integers corresponding to each sample).
        """
        # copy of the data in tokenized format
        self.tokenized_data: List[List[int]] = []

        for raw_text in self.data:
            # tokenize sample
            tokenizer_output = tokenizer(
                text=raw_text,
                padding="max_length",
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            tokenizer_output.input_ids, labels = collator.torch_mask_tokens(
                tokenizer_output.input_ids
            )

            self.tokenized_data.append(dict(**tokenizer_output, labels=labels))

        for i, item in enumerate(self.tokenized_data):
            self.tokenized_data[i] = {k: v.squeeze() for k, v in item.items()}

        return self.tokenized_data

    def create_token_hist(
        self,
        token_counts: List[int] = [],
    ) -> matplotlib.figure.Figure:
        """
        Calculates the number of utterances in each sample and generates a histogram.

        Utterance counts are determined by splitting each transcript on whitespace and
        calculating the length of the resulting list.

        TODO: add flexibility for plot types.
        TODO: issue with figures (figures/axes need to be cleared between runs)

        Arguments:
        ----------
        `utterance_counts`: (Optional) `list` of `ints` for precalculated utterance counts.

        `plot_type`: (Optional) `str` type of plot tool to use to create the histogram.
        Can be either `"seaborn"` or `"matplotlib"`. Defaults to `"seaborn"`.

        Returns:
        --------
        Either a `matplotlib.pyplot.Figure` or `seaborn.object.Plot` instance, depending on the value of `plot_type`.
        """
        # Clear figure and axes
        plt.clf(), plt.cla()
        # check if manifest data has been generated, parse transcripts and generate
        # manifest data if not
        if len(self.data) == 0:
            self.parse_transcripts()

        # check if utterance counts (optional arg) has been provided, calculate utterance
        # counts from transcriptions
        if len(token_counts) == 0:
            for data in self.data:
                words = data.split(" ")
                token_counts.append(len(words))

        histogram = plt.hist(token_counts)
        plt.xlabel("Bins")
        plt.ylabel("Token Counts")
        plt.title(f"Tokens per Sample in {self.name}")

        return histogram

    def token_freq_analysis(self, normalize=False) -> Dict[str, Union[int, List]]:
        """
        Perform a token frequency analysis on the dataset (number of occurrences of each token throughout the dataset).

        Arguments:
        ----------
        `normalize`: (optional)`bool`, whether to normalize values such that all frequencies add to 1.

        Returns:
        --------
        `token_freqs`: `dict` with tokens and number of occurrences of those tokens throughout the dataset.
        If `normalize` is set to `True` a dictionary with tokens corresponding to a list is returned e.g.
        ```
        {
            "token": [24, 0.0486] # token with corresponding occurences and frequencies
        }
        ```

        """
        if len(self.data) == 0:
            self.parse_transcripts()

        # tokens corresponding to occurences/frequencies
        token_freqs = {}

        for sample in self.data:
            for token in sample.split():
                if token in token_freqs.keys():
                    # increment occurences if there is already an entry
                    token_freqs[token] += 1
                else:
                    # add entry if one does not exist already
                    token_freqs[token] = 1

        if normalize:
            # convert from token occureces to frequencies: (# of token occurences / # of tokens)
            num_tokens = len(token_freqs)
            for token, freq in token_freqs.items():
                token_freqs[token] = [freq, float(freq) / num_tokens]

        return token_freqs

    def dump_corpus(
        self, outfile: str, make_dirs: bool = True, return_list: bool = False
    ) -> Union[None, List[str]]:
        """
        Dump input data paths, labels, and metadata to `outfile` in NeMo manifest format.

        Arguments:
        ----------
        `outfile`: `str`, output path

        `make_dirs`: (optional) `bool`, whether to make nonexistent parent directories
        in `outfile`. Defaults to `True`.

        `return_list`: Returns a list of strings instead of creating and dumping to a file.

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

        if return_list:
            return self.data

    def concat(self, child_dataset: "Data"):
        """
        Concatenates data classes/sets. Extends the data array of parent object to
        include data from child object. Also updates any relevant common metadata
        fields.
        """
        self.data.extend(child_dataset.data)
        self.dataset_name = f"{self.name} + {child_dataset.name}"

    def dump_info(self) -> Dict[str, Union[str, int, float]]:
        """
        Returns all relevant dataset statistics in a dictionary.

        The following fields are populated:
        `'dataset_name'` - Name of the dataset, `str`
        `'samples'` - Numher of samples in the dataset, `int`
        `'total_tokens'` - Total number of tokens present in the dataset, found across all samples, `int`
        `'unique_tokens'` - Number of unique tokens in the dataset, `int`
        """
        return {
            "dataset_name": self.name,
            "samples": self.num_samples,
            "total_tokens": self.total_tokens,
            "unique_tokens": self.unique_tokens,
        }

    def token_ratio(self, as_tuple=False) -> Union[float, Tuple[int, int]]:
        """
        The ratio of unique tokens to the total number of tokens.

        :param as_tuple: Defaults to False. If True, returns a tuple
        of the ratio of unique tokens to total tokens in the order [unique, total].
        """
        unique_tokens = self.unique_tokens
        total_tokens = self.total_tokens

        if as_tuple:
            return tuple([unique_tokens, total_tokens])
        else:
            return unique_tokens / total_tokens

    def summary(self, printout=True):
        """
        Print out (or return a formatted string) with available statistics about this dataset.

        :param printout: Defaults to True. If False, returns the formatting string that would
        have been printed to the terminal.
        """
        dataset_summary = f"""
Name: {self.name}
Samples: {self.num_samples}
Mean Sequence Length: {self.average_sequence_length}
Number of Tokens: {self.total_tokens}
Number of Unique Tokens: {self.unique_tokens}
Ratio of unique tokens to the total number of tokens: {self.token_ratio()}, {self.token_ratio(as_tuple=True)}
"""
        if printout:
            print(dataset_summary)
        else:
            return dataset_summary

    @classmethod
    def from_corpus(cls: "Data", corpus_path: str, random_seed: int = 1) -> "Data":
        """
        Loads a dataset from a standard corpus file e.g. `corpus.txt`. These are typically
        text files where each line in the file corresponds to a single sample i.e. samples
        separated by newlines.

        Arguments:
        ----------
        `corpus_path`: `str`, path to the corpus file

        `random_seed`: `int`, random seed to initialize the class with (default to `1`)

        Returns:
        --------
        An instance of this class (`Data`) initialized with data from `corpus_path`
        """
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus path not found: '{corpus_path}'")

        # initialize object from class
        data = cls(random_seed=random_seed)

        # read corpus and add data to object
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data.data.append(line)

        return data

    @property
    def name(self) -> str:
        return self.dataset_name

    @property
    def num_samples(self) -> int:
        """
        Number of samples in the dataset
        """
        return len(self.data)

    @property
    def total_tokens(self) -> int:
        """
        Total number of tokens in the transcripts of the dataset (labels)
        """
        num_tokens = 0

        for item in self.data:
            # very simple way to do this, should use a tokenizer passed to this class
            tokens = item.split(" ")
            num_tokens += len(tokens)

        return num_tokens

    @property
    def unique_tokens(self) -> int:
        """
        Number of unique tokens in the dataset. Repeating tokens are removed
        from this total
        """
        unique_tokens = []

        # simplest way to do this again; should consider more sophisticated approach
        for item in self.data:
            tokens = item.split(" ")

            for token in tokens:
                if token not in unique_tokens:
                    unique_tokens.append(token)

        return len(unique_tokens)

    @property
    def average_sequence_length(self) -> int:
        """
        Average number of tokens per sequence in the dataset. Calculated as
        the sum of the sequence lengths divided by the number of samples.
        """
        sum_of_seq_lens = 0

        for item in self.data:
            sum_of_seq_lens += len(item.split(" "))

        return sum_of_seq_lens / self.num_samples


def get_train_test_split(
    data: Data,
    split_ratio: float = 0.7,
    valid_split_ratio: float = 0.1,
    shuffle: bool = False,
    random_seed: int = 1,
) -> Tuple[Data, Data, Data]:
    """
    Utility function to split a dataset, `Data` into train and test subsets.

    TODO: add functionality for validation set

    Arguments:
    ----------
    `data`: (required) the dataset to split. Expected type: `Data`.

    `split_ratio`: (optional) ratio of train data to test data. Defaults to `0.7`.

    `shuffle`: (optional) whether to shuffle the data before splitting into subsets. Defaults to `False`.

    `random_seed`: (optional)Set the seed of the random function. Use this when the results need to be
    reproducable. Only has an effect on the output when `shuffle` is `True`.

    Returns:
    --------
    A tuple of train, validation, and test subsets i.e. (train, valid, test)

    """
    if split_ratio >= 1.0 or split_ratio <= 0.0:
        raise ValueError(
            f"The train/test split ratio needs to be between 0 and 1 (exclusive) got: '{split_ratio}' instead"
        )

    if valid_split_ratio >= 1.0 or valid_split_ratio <= 0.0:
        raise ValueError(
            f"The validation split ratio needs to be between 0 and 1 (exclusive) got: '{valid_split_ratio}' instead"
        )

    data_length = len(data.data)
    train_size = int(split_ratio * data_length)
    valid_size = int(valid_split_ratio * train_size)

    # getting a copy of the data before any modification happens
    # shuffle happens 'in-place', we don't want to unintentionally
    # modify any data
    data = deepcopy(data.data)

    if shuffle:
        # numpy rng instance (w/ random seed set)
        random = default_rng(random_seed)
        # shuffle the data
        random.shuffle(data)

    # just using slicing to split up train and test sets
    train_set = data[: train_size - valid_size]
    valid_set = data[train_size - valid_size : train_size]
    test_set = data[train_size:]

    train = Data(random_seed=random_seed)
    valid = Data(random_seed=random_seed)
    test = Data(random_seed=random_seed)

    train.data = train_set
    valid.data = valid_set
    test.data = test_set

    return train, valid, test


class PLDataLoader(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, preprocess_fn: Callable = None):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if preprocess_fn is not None:
            self.preprocess = preprocess_fn

    def train_dataloader(self):
        if self.preprocess:
            return self.preprocess(self.train_dataset)
        else:
            return DataLoader(self.train_dataset)

    def val_dataloader(self):
        if self.preprocess:
            return self.preprocess(self.val_dataset)
        else:
            return DataLoader(self.val_dataset)
