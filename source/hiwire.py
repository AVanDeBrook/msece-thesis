import os
import glob
from typing import *
from pathlib import Path
from data import Data, hiwireutils


class HIWIREData(Data):
    """
    The following attributes are defined in this class due to the specific formatting of
    the HIWIRE dataset.

    This dataset can be obtained here:
        https://catalogue.elra.info/en-us/repository/browse/ELRA-S0293/

    Attributes:
    -----------
    `_transcript_glob`: list of aths of transcript files that contain labels for the audio data in the dataset.

    """

    _transcript_glob: List[str]

    def __init__(self, data_root: str, **kwargs):
        super(HIWIREData, self).__init__(data_root, **kwargs)

        # glob search string to get list of transcript paths (absolute paths)
        search_string = os.path.join(data_root, "**/list*.txt")
        HIWIREData._transcript_glob = glob.glob(search_string, recursive=True)

    def parse_transcripts(self) -> Dict[str, Union[str, float]]:
        # ensure there are actually transcript paths present
        assert len(self._transcript_glob) != 0, "No transcript paths found."

        manifest_data = []

        for path in self._transcript_glob:
            # parse returns a list of dictionaries
            manifest_data.extend(hiwireutils.parse(path))

        # save a copy of the manifest data to the class attribute before returning
        HIWIREData._manifest_data = manifest_data
        return manifest_data

    @property
    def name(self):
        return "HIWIRE"
