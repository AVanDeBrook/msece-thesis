import os
import glob
import re
from data import Data
from typing import *


class ATCOSimData(Data):
    """
    This class describes the format of the Air Traffic Control Simulation Corpus
    and defines functions for parsing the data into a common format for data analysis.

    This dataset is described in more depth and can be obtained here:
        https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html

    The data is shipped in iso (disk) format. To extract, mount the disk onto the file system and copy the data
    into another directory.
    ```
    mount -t iso9660 -o loop atcosim.iso atcosimmount
    cp -r atcosimmount .
    ```
    """

    transcription_corrections = [
        ("kil0", "kilo"),
        ("ai", "air"),
        ("airr", "air")
    ]

    def __init__(self, data_root: str, **kwargs):
        super(ATCOSimData, self).__init__(data_root, **kwargs)
        # TODO: update regex
        self.text_glob = glob.glob(
            os.path.join(data_root, "txtdata/**/*.txt"), recursive=True
        )

        # at the moment this is easier than updating the regex to exclude this specific file
        # TODO: update regex
        wordlist_path = os.path.join(data_root, "txtdata/wordlist.txt")
        if os.path.exists(wordlist_path):
            self.text_glob.remove(wordlist_path)

    def parse_transcripts(self) -> List[str]:
        data = []

        # regular expressions for removing transcript annotations
        xml_tag = re.compile(r"(<[A-Z]+>|</[A-Z]+>)")
        annotation_tag = re.compile(r"(\[[A-Z]+\])")
        special_chars = re.compile(r"[=~@]")
        hesitation_tokens = re.compile(r"(ah|hm|ahm|yeah|aha|nah|ohh)")
        non_english_tags = re.compile(r"(<FL>.</FL>)")

        for file in self.text_glob:
            # read data from file
            with open(file, "r") as f:
                text = "".join([t.strip() for t in f.readlines()])

            # skip non-english samples
            if non_english_tags.match(text):
                continue

            # remove transcript annotations
            text = xml_tag.sub("", text)
            text = annotation_tag.sub("", text)
            text = special_chars.sub("", text)
            text = hesitation_tokens.sub("", text)

            # lower case, remove whitespace
            text = text.lower().strip()

            # correct identified typos, see `transcription_crrections` for
            # the full list
            for typo, correction in self.transcription_corrections:
                if typo in text:
                    text = text.replace(typo, correction)

            # some transcripts are empty after removing transcriber
            # annotations
            if len(text) > 0:
                data.append(text)


        ATCOSimData.data = data
        return data

    @property
    def name(self):
        return "ATCO"
