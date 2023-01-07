import os
import glob
import subprocess
from typing import *
from data import Data
from data import atccutils
from pathlib import Path
from pprint import pprint


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

    # _audio_glob: List[str]
    # _transcript_glob: List[str]

    def __init__(self, data_root: str, **kwargs):
        super(ATCCompleteData, self).__init__(data_root, **kwargs)

        # search strings for sphere and wav audio files
        sphere_glob_string = os.path.join(data_root, "**/data/audio/*.sph")
        wav_glob_string = os.path.join(data_root, "**/data/audio/*.wav")

        # file path globs for sphere and wav files
        sphere_glob = glob.glob(sphere_glob_string, recursive=True)
        self._audio_glob = glob.glob(wav_glob_string, recursive=True)

        # if there are only sphere files present, convert to wav
        if len(sphere_glob) > 0 and len(self._audio_glob) == 0:
            self._convert_audio_files(sphere_glob)
            self._audio_glob = glob.glob(wav_glob_string, recursive=True)

        # paths to transcript files
        transcript_glob_string = os.path.join(data_root, "**/data/transcripts/*.txt")
        self._transcript_glob = glob.glob(transcript_glob_string, recursive=True)

    def _convert_audio_files(self, audio_paths: List[str]):
        """
        Converts files from NIST Sphere format to Microsoft WAV format. **Requires ffmpeg to be installed**.

        Arguments:
        ----------
        `audio_paths`: paths to audio files to convert.

        Returns:
        --------
        None
        """
        # ffmpeg required to perform the conversion from sphere to wav
        if os.system("command -v ffmpeg") != 0:
            raise Exception(
                "Cannot find ffmpeg. Please install ffmpeg (and add to the system path, if applicable) and rerun the script."
            )

        # convert each audio file in sphere format to wav format
        for file in audio_paths:
            if os.path.exists(file):
                ffmpeg_command = [
                    "ffmpeg",
                    # force ffmpeg to assume yes to all y/n inputs
                    "-y",
                    "-i",
                    file,
                    # resample from input format to 16kHz where applicable
                    "-ar",
                    "16000",
                    # output file with same name and wav extension
                    file.replace("sph", "wav"),
                ]
                subprocess.run(ffmpeg_command)

    def parse_transcripts(self) -> List[Dict[str, Union[str, float]]]:
        """
        Parse data indices in transcript files into dictionary objects with required info
        to be compatible with NeMo's manifest format.

        Returns:
        --------
        A list of dictionary objects.
        """
        data = []

        for text, audio in zip(self._transcript_glob, self._audio_glob):
            # need absolute file path for compliance with NeMo manifest format
            audio = Path(audio).absolute()

            # parse transcript file (returns a list of dictionary objects where each
            # object corresponds to each Lisp list in the transcript file)
            with open(text, "r", encoding="utf-8") as f:
                transcript_data = atccutils.parse(f.readlines())

            # filter out tape-header, tape-tail, and comment blocks (and any other
            # extraneous info that could cause KeyErrors)
            for datum in transcript_data:
                if "TEXT" in datum.keys():
                    data.append(datum["TEXT"])

        # save manifest data to class attribute before returning
        ATCCompleteData.data = data
        return data

    @property
    def name(self):
        return "Air Traffic Control Complete"
