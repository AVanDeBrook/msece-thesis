"""
This file contains the functions necessary to preprocess
"""
import os
import json
import glob
from pprint import pprint
import librosa
from pathlib import Path
from typing import List, Union


d2w = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def _reformat_labels(labels: List[str]) -> List[str]:
    output = []
    # modified from https://github.com/eraus/pfiga_sperega/blob/main/1pj/nemo_asr_development/alterManifest.ipynb
    for label in labels:
        words = label.split()
        for i, word in enumerate(words):
            if word.isupper():
                word = " ".join([char for char in word])
            # word = word.lower()
            words[i] = word
        temp = " ".join(words)

        debug = False
        for char in ["-", "/"]:
            if "/" in temp:
                print(temp)
                debug = True
            temp = temp.replace(char, " ")
            if debug:
                print(temp)
                debug = False

        for num in d2w.keys():
            if num in temp:
                temp = temp.replace(num, d2w[num])
        output.append(temp)
    return output


def parse(transcript_path: str) -> List[str]:
    # using pathlib to manage files/paths is easier in this case
    transcript_path = Path(transcript_path).absolute()
    data = []

    # wav files correspond to each line in the transcript file
    # sorting by name ensures transcripts align to audio files 1-to-1
    wav_files = sorted(
        [
            file.absolute()
            for file in transcript_path.parent.iterdir()
            if file.suffix == ".wav"
        ],
        key=lambda x: x.name,
    )

    # get labels from transcript file and reformat
    with transcript_path.open("r") as f:
        # skip first line (just lists how many labels are in the file)
        labels = f.readlines()[1:]
        # remove end of the line of each transcript
        labels = [label[: label.find("(")].strip() for label in labels]
        labels = _reformat_labels(labels)

    # construct manifest info, save to data list as a dictionary
    for wav, label in zip(wav_files, labels):
        data.append(
            {
                "audio_filepath": str(wav),
                "text": label,
                "duration": float(librosa.get_duration(filename=str(wav))),
            }
        )

    return data
