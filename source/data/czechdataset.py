import os
import glob
import re
from typing import *
from data import Data
from xml.etree import ElementTree
from tokenizers.normalizers import NFD


class ZCUCZATCDataset(Data):
    """
    Data is organized into `trs` files (follows an XML standard). Details in `~parse_transcripts` function.

    Dataset can be obtained here: https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCA1-0
    """

    _transcript_corrections = [
        ("+", "-")
    ]

    def __init__(self, data_root: str, **kwargs):
        super(ZCUCZATCDataset, self).__init__(dataset_name="ZCU CZ ATC", **kwargs)
        self.transcript_paths = glob.glob(os.path.join(data_root, "*.trs"))

        assert (
            len(self.transcript_paths) != 0
        ), f"Cannot find transcripts in data_root: {data_root}"

    def parse_transcripts(self) -> List[str]:
        """
        Since the transcript files correspond to wav files for ASR tasks, the transcriptions are organized into
        <Sync> elements with time attributes e.g. <Sync time="1.900"/>. In this case the time attribute is ignored
        and only the text is extracted. The node hierarchy is as follows:
            * <Trans> -- root node
            * <Episode> -- empty tag (organizational?)
            * <Section> -- metadata, similar to <Turn>
            * <Turn> -- contains metadata about the duration of the audio file (attributes), serves as parent for <Sync> nodes
            * <Sync> -- holds the time (attribute) and text info (tail)

        There are also transcriber annotations present in the text, usually following a form similar to other transcripts
        for example:
            * [air]
            * [ground]
            * [unintelligible]
        """
        data = []

        annotation_tag = re.compile(r"(\[[A-Za-z_\|\?]+\])")
        nonstandard_pronunciation = re.compile(
            r"\((?P<effective_transcript>[\w\d\s\+]+)\((?P<phonetic_transcript>[\w\d\s]+)\)\)"
        )
        normalizer = NFD()

        for transcript_path in self.transcript_paths:
            doc_data = []
            try:
                # root node: <Trans>
                document = ElementTree.parse(transcript_path).getroot()
            except ElementTree.ParseError as e:
                # because not all transcripts conform to the given format
                with open(transcript_path, "a") as dumb_path:
                    # there is a single file that is missing the closing tags on all nodes
                    # hard-coding the fix here, so I can forget about it
                    dumb_path.write("</Turn>\n</Section>\n</Episode>\n</Trans>\n")
                # parse the document again
                document = ElementTree.parse(transcript_path).getroot()

            # find <Sync> tags, extract text, reformat/clean
            for sync_node in document.iterfind(
                ".//Sync"
            ):  # searches all subelements for Sync nodes
                text: str = annotation_tag.sub("", sync_node.tail).strip()

                for nonstandard_match in nonstandard_pronunciation.finditer(text):
                    if nonstandard_match is not None:
                        text = text.replace(
                            nonstandard_match.group(0), nonstandard_match.group(1)
                        )

                # typos in transcripts
                for typo, correction in self._transcript_corrections:
                    if typo in text:
                        text = text.replace(typo, correction)

                # ".." corresponds to silent segments, "" can occur when the transcript is made up
                # of only transcriber annotations
                if text != ".." and text != "":
                    text = text.translate(
                        str.maketrans(
                            {
                                "0": "zero",
                                "1": "one",
                                "2": "two",
                                "3": "tree",
                                "4": "four",
                                "5": "fife",
                                "6": "six",
                                "7": "seven",
                                "8": "eight",
                                "9": "niner",
                                ".": "dot",
                            }
                        )
                    )

                    # unicode string normalization, gets rid of some but not all foreign and uncommon characters
                    text = normalizer.normalize_str(text)
                    text = re.sub(r"\s{2,}", " ", text)
                    doc_data.append(text.strip())

            # check to make sure useful samples were found in the document
            if len(doc_data) != 0:
                data.extend(doc_data)

        self.data = data
        return data
