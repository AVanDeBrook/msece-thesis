import json
import os
import logging
from typing import *

import matplotlib.pyplot as plt
import numpy as np
from data import ATCCompleteData, ATCO2SimData, ATCOSimData, Data, ZCUATCDataset

# root dataset paths corresponding to data analysis classes
datasets: Dict[str, Data] = {
    # TODO: find a way to sync file paths across computers (shell/env var, config file?)
    "/home/students/vandebra/programming/thesis_data/atc0_comp": ATCCompleteData,
    "/home/students/vandebra/programming/thesis_data/atcosim/": ATCOSimData,
    "/home/students/vandebra/programming/thesis_data/ATCO2-ASRdataset-v1_beta": ATCO2SimData,
    # TODO: this dataset has strange formats for transcriptions, need to do a lot of
    # work to clean and reformat them. Disabling this dataset for now
    # "/home/students/vandebra/programming/thesis_data/ZCU_CZ_ATC": ZCUATCDataset,
}

if __name__ == "__main__":
    RANDOM_SEED: int = 1
    # collection of dataset stats. using a dictionary so it's easier to dump to JSON or YAML later
    dataset_info = {"dataset_info": []}
    # collection of initialized `Data` classes so they don't get gc'd and for concatenating everything
    # after all data has been parsed/collected
    data_objects: List[Data] = []

    plt.style.use("ggplot")

    # initializes each implementing class with its data which is specified by `root_path`
    # see `datasets` and implementing classes for more details
    for root_path, data_class in datasets.items():
        # initialize class with root path and random seed
        data_analysis: Data = data_class(data_root=root_path, random_seed=RANDOM_SEED)

        # parse dataset info, along with some printout so the user has some idea of what's
        # happening
        print(f"Parsing transcripts for '{data_analysis.name}'", end="...")
        data_analysis.parse_transcripts()
        print("Done")

        # printing some stats to stdout just to confirm things worked correctly
        print(
            f"Found {data_analysis.num_samples} samples with {data_analysis.unique_tokens} "
            f"unique tokens and {data_analysis.total_tokens} tokens in total for '{data_analysis.name}'"
        )
        # dump corpora to a standard corpus.txt style file, we'll have one for each dataset and
        # one for everything combined together
        data_analysis.dump_corpus(f"corpora/{data_analysis.name.replace(' ', '_')}")

        dataset_info["dataset_info"].append(data_analysis.dump_info())
        data_objects.append(data_analysis)

    # write stats to a json file
    with open("manifests/dataset_stats.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(dataset_info, indent=1))

    # concatenate everything to first object
    for o in data_objects[1:]:
        data_objects[0].concat(o)

    # combined corpus/corpora
    data_objects[0].dump_corpus(f"corpora/all_copus.txt")
