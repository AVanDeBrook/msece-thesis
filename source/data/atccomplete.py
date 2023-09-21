import os
import glob
import re
from typing import *
from data import Data, atccutils


class ATCCompleteData(Data):
    """
    This class defines the data format of the Air Traffic Control Complete dataset and
    provides functions for parsing the data into a common format for automatic data
    analysis (see ~Data).

    This dataset is described in depth and can be obtained here:
        https://catalog.ldc.upenn.edu/LDC94S14A

    The following attributes are defined in this class due to the way the Air Traffic
    Control Complete dataset is formatted and distributed.

    Attributes:
    -----------
    `_audio_glob`: list of paths to audio files in the dataset.

    `_transcript_glob`: list of paths to transcript files that correspond to the audio
    files in the dataset. Transcripts are formatted as Lisp lists, each list corresponds
    to one sample in the data i.e. one transmission.
    """

    # there are a lot of typos in this dataset; this is the running list with corrections
    # list of (typo, correction) pairs (tuples)
    _transcript_corrections = [
        ("0h", "oh"),
        ("0r", "or"),
        ("8'll", "i'll"),
        ("kil0", "kilo"),
        ("altimeter;;;'s", "altimeter's"),
        ("bye]", "bye"),
        (" -", ""),
        (" 3 ", "three"),
        ("1347.85706", "one three four seven dot eight five seven zero six"),
        ("four]", "four"),
        # flight number metadata somehow made it into some of the transcripts
        ("swift61", "swift six one"),
        ("aal891", "american eight ninety one"),
        # repeated words/hesitations
        ("ai", ""),
        ("cir-", "cir+"),
        ("cli-", "cli+"),
        ("rport", "airport"),
        ("rcraft", "aircraft"),
        ("mntn", "maintain"),
        ("tornado's", "tornadoes"),
    ]

    def __init__(self, data_root: str, **kwargs):
        super(ATCCompleteData, self).__init__(
            dataset_name="Air Traffic Control Complete", **kwargs
        )
        transcript_glob_string = os.path.join(data_root, "**/data/transcripts/*.txt")
        self._transcript_glob = glob.glob(transcript_glob_string, recursive=True)

    def parse_transcripts(self) -> List[Dict[str, Union[str, float]]]:
        """
        Parse data indices in transcript files into dictionary objects with required info
        to be compatible with NeMo's manifest format.

        Returns:
        --------
        A list of dictionary objects.
        """
        data = []

        sounds_like_tag = re.compile(r"(\[sounds like(?P<guess>[A-Za-z\s\d]+)\]*)")
        annotation = re.compile(r"(\[[A-Za-z\s]+\])")

        for text in self._transcript_glob:
            # parse transcript file (returns a list of dictionary objects where each
            # object corresponds to each Lisp list in the transcript file)
            with open(text, "r", encoding="utf-8") as f:
                transcript_data = atccutils.parse(f.readlines())

            # filter out tape-header, tape-tail, and comment blocks (and any other
            # extraneous info that could cause KeyErrors)
            for datum in transcript_data:
                if "TEXT" in datum.keys():
                    text: str = datum["TEXT"].lower()
                    # line breaks
                    text = "\n".join([t.strip() for t in text.split("//")])
                    text = "\n".join([t.strip() for t in text.split("/")])

                    # transcriber guesses
                    sounds_like_match = sounds_like_tag.match(text)
                    if sounds_like_match is not None:
                        text = sounds_like_match["guess"].strip()

                    # typos in transcripts
                    for typo, correction in self._transcript_corrections:
                        if typo in text:
                            text = text.replace(typo, correction)

                    # remove transcriber annotations
                    text = annotation.sub("", text)

                    if text.strip() != "":
                        text = re.sub(r"\s{2,}", " ", text)
                        data.append(text.strip())

        # save manifest data to class attribute before returning
        self.data = data
        return data
