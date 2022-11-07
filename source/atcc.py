import os
import glob
import subprocess
from typing import *
from data import Data


class ATCCData(Data):
    _audio_glob: List[str]
    _transcript_glob: List[str]


    def __init__(self, data_root: str):
        super(Data, self).__init__(data_root)

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

        # required fields to be extracted from dataset transcripts
        self._required_fields = [
            "TEXT",
            "TIMES",
        ]

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
            raise Exception("Cannot find ffmpeg. Please install ffmpeg (and add to the system path, if applicable) and rerun the script.")

        for file in audio_paths:
            if os.path.exists(file):
                ffmpeg_command = ["ffmpeg", "-y", "-i", file, file.replace("sph", "wav")]
                subprocess.run(ffmpeg_command)



    def parse_transcripts(self) -> Dict[str, Union[str, float]]:

