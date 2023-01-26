import os
import glob
import re
from typing import *
from data import Data
from xml.etree import ElementTree


class ZCUATCDataset(Data):
    """
    Data is organized into `trs` files (follows an XML standard). Details in `~parse_transcripts` function.

    Dataset can be obtained here: https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCA1-0
    """

    def __init__(self, data_root: str, **kwargs):
        super(ZCUATCDataset, self).__init__(data_root, **kwargs)

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

        for path in self.transcript_paths:
            try:
                # root node: <Trans>
                document = ElementTree.parse(path).getroot()
            except ElementTree.ParseError as e:
                # because not all transcripts conform to the given format
                with open(path, "a") as dumb_path:
                    # there is a single file that is missing the closing tags on all nodes
                    # hard-coding the fix here, so I can forget about it
                    dumb_path.write("</Turn>\n</Section>\n</Episode>\n</Trans>\n")
                # parse the document again
                document = ElementTree.parse(path).getroot()

            # find <Sync> tags, extract text, reformat/clean
            for sync_node in document.iterfind(
                ".//Sync"
            ):  # searches all subelements for Sync nodes
                assert sync_node.tag == "Sync"
                assert len(sync_node.tail) != 0

                text = annotation_tag.sub("", sync_node.tail).strip()
                # ".." corresponds to silent segments, "" can occur when the transcript is made up
                # of only transcriber annotations
                if text != ".." and text != "":
                    data.append(text)

        ZCUATCDataset.data = data
        return data

    @property
    def name(self) -> str:
        return "ZCU ATC"
