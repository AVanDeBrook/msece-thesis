import os
import glob
import re
import json
from pprint import pprint
from typing import *
from typing import List
from data import Data, atccutils


class TransmissionTurn(object):
    def __init__(
        self,
        start_time: float,
        end_time: float,
        text: str,
        to_key: str,
        from_key: str,
    ) -> "TransmissionTurn":
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
        self.to_key = to_key
        self.from_key = from_key

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_dict(self) -> Dict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "to_key": self.to_key,
            "from_key": self.from_key,
        }

    def __repr__(self) -> str:
        return self.to_json()


class TransmissionGroup(object):
    def __init__(
        self,
        first_transmission: TransmissionTurn,
        transmissions: List[TransmissionTurn],
        file: str,
        cls_token="[CLS]",
        sep_token="[SEP]",
    ) -> "TransmissionGroup":
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.first_transmission = first_transmission
        self.transmissions = transmissions
        self.file = file

    def to_json(self) -> str:
        return json.dumps(
            {
                "cls_token": self.cls_token,
                "sep_token": self.sep_token,
                "file": self.file,
                "first_transmission": self.first_transmission.to_dict(),
                "transmissions": [t.to_dict() for t in self.transmissions],
            }
        )

    def __repr__(self) -> str:
        return self.to_json()


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
        ("1347.85706", "one three four seven decimal eight five seven zero six"),
        ("four]", "four"),
        # flight number metadata somehow made it into some of the transcripts
        ("swift61", "swift six one"),
        ("aal891", "american eight ninety one"),
        # repeated words/hesitations
        ("ai", ""),
        ("rport", "airport"),
        ("rcraft", "aircraft"),
        ("rborne", "airborne"),
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
        assert len(self._transcript_glob) != 0
        data = []

        sounds_like_tag = re.compile(r"(\[sounds like(?P<guess>[A-Za-z\s\d]+)\]*)")
        annotation = re.compile(r"(\[[A-Za-z\s]+\])")

        for text in self._transcript_glob:
            # parse transcript file (returns a list of dictionary objects where each
            # object corresponds to each Lisp list in the transcript file)
            with open(text, "r", encoding="utf-8") as f:
                transcript_data = atccutils.parse(f.readlines(), text)

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

    def parse_transcripts_for_nsp(self) -> List[str]:
        assert len(self._transcript_glob) != 0
        file_groupings = {}
        transmission_groups = []

        sounds_like_tag = re.compile(r"(\[sounds like(?P<guess>[A-Za-z\s\d]+)\]*)")
        annotation = re.compile(r"(\[[A-Za-z\s]+\])")
        callsign_expression = re.compile(
            r"([A-Z]{3,4}[0-9]{2,4})|([A-Z][0-9][A-Z0-9]+)"
        )

        # read files
        for transcript_file in self._transcript_glob:
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript_data = atccutils.parse(f.readlines(), transcript_file)

            for datum in transcript_data:
                # copied from above for now
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

                    datum["TEXT"] = text

                    if transcript_file in file_groupings:
                        # add to list if group exists
                        file_groupings[transcript_file].append(datum)
                    else:
                        # create new key and add to list if it doesn't
                        file_groupings[transcript_file] = [datum]

            # sort objects in each grouping by start time
            file_groupings[transcript_file] = sorted(
                file_groupings[transcript_file], key=lambda item: item["TIMES"]["start"]
            )

            # do this by file:
            for group in file_groupings.keys():
                # start with first FROM marker
                for i, data in enumerate(file_groupings[group]):
                    # unfortunately, this is the simplest way to do this (with a break)
                    if callsign_expression.match(data["FROM"]):
                        from_marker = data["FROM"]
                        # print(f"First from marker: {data}")
                        break

                # iterate through all matching TO markers in file
                candidate_pairs = []
                for data in file_groupings[group][i:]:
                    # create candidate pairs from these matches
                    if data["TO"] == from_marker:
                        candidate_pairs.append(data)
                    # trim based on start/end times and time between transmissions
                # pprint(candidate_pairs)

                transmission_group = []
                for pair in candidate_pairs:
                    transmission_group.append(
                        TransmissionTurn(
                            start_time=pair["TIMES"]["start"],
                            end_time=pair["TIMES"]["end"],
                            text=pair["TEXT"],
                            to_key=pair["TO"],
                            from_key=pair["FROM"],
                        )
                    )

                first_transmission = file_groupings[group][i]
                first_transmission = TransmissionTurn(
                    start_time=first_transmission["TIMES"]["start"],
                    end_time=first_transmission["TIMES"]["end"],
                    text=first_transmission["TEXT"],
                    to_key=first_transmission["TO"],
                    from_key=first_transmission["FROM"],
                )

                transmission_groups.append(
                    TransmissionGroup(
                        first_transmission,
                        transmission_group,
                        file_groupings[group][i]["FILE"],
                    )
                )

        pprint(transmission_groups)
