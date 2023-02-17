import json
import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
from data import ATCCompleteData, ATCO2SimData, ATCOSimData, Data, ZCUATCDataset

RANDOM_SEED: int = 1

# root dataset paths corresponding to data analysis classes
datasets: Dict[str, Data] = {
    # TODO: find a way to sync file paths across computers (shell/env var, config file?)
    "/home/avandebrook/thesis_data/atc0_comp": ATCCompleteData,
    "/home/avandebrook/thesis_data/atcosim/": ATCOSimData,
    "/home/avandebrook/thesis_data/ATCO2-ASRdataset-v1_beta": ATCO2SimData,
    "/home/avandebrook/thesis_data/ZCU_CZ_ATC": ZCUATCDataset,
}

if __name__ == "__main__":
    plt.style.use("ggplot")

    os.makedirs("corpora", exist_ok=True)
    with open("corpora/all_corpora.json", "w", encoding="utf-8") as all_corpus:
        with open("corpora/dataset_stats.json", "w", encoding="utf-8") as dataset_stats:
            for root_path, data_class in datasets.items():
                # create and initialize dataset
                dataset: Data = data_class(data_root=root_path, random_seed=RANDOM_SEED)
                print(f"Parsing data in {dataset.name}", end="...")
                dataset.parse_transcripts()
                print("Done")

                # dump data to txt file (one sample per line; see function docstring)
                outfile_path = f"corpora/{dataset.name.replace(' ', '_')}.txt"
                print(f"Dumping dataset to '{outfile_path}'", end="...")
                data = dataset.dump_corpus(outfile_path, return_list=True)
                print("Done")

                # combine all samples from all dataset into one file
                for sample in data:
                    all_corpus.write(
                        json.dumps({"dataset_name": dataset.name, "text": sample})
                    )
                    all_corpus.write("\n")

                # collect dataset statistics, store them in one file
                dataset_stats.write(
                    json.dumps(
                        {
                            "name": dataset.name,
                            "num_samples": dataset.num_samples,
                            "total_tokens": dataset.total_tokens,
                            "unique_tokens": dataset.total_tokens,
                        }
                    )
                )
                dataset_stats.write("\n")
