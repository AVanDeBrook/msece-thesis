from typing import *
from pprint import pprint
from data import Data, ATCCompleteData, ATCOSimData, ATCO2SimData, ZCUATCDataset
from matplotlib.figure import Figure
from seaborn.objects import Plot
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    # random seed initialization (passed to data classes) for reproducible randomness
    # in results
    RANDOM_SEED: int = 1
    plt.style.use("ggplot")
    os.makedirs("corpora", exist_ok=True)

    # root dataset paths corresponding to data analysis classes
    datasets: Dict[str, Data] = {
        # TODO: find a way to sync file paths across computers (shell/env var, config file?)
        "/home/avandebrook/thesis_data/atc0_comp": ATCCompleteData,
        "/home/avandebrook/thesis_data/atcosim/": ATCOSimData,
        "/home/avandebrook/thesis_data/ATCO2-ASRdataset-v1_beta": ATCO2SimData,
        "/home/avandebrook/thesis_data/ZCU_CZ_ATC": ZCUATCDataset,
    }

    # the following two values are running totals across the datasets
    num_unique_tokens = 0
    num_tokens = 0

    for root_path, data_class in datasets.items():
        data_analysis: Data = data_class(data_root=root_path, random_seed=RANDOM_SEED)

        # parse transcripts in dataset
        print(f"Parsing transcripts for '{data_analysis.name}'", end="...")
        data_analysis.parse_transcripts()
        print("Done")

        token_freq = data_analysis.token_freq_analysis(normalize=True)
        # unique tokens found
        num_unique_tokens += len(token_freq.keys())
        # absolute number of tokens
        num_tokens += np.sum([count[0] for count in token_freq.values()])

        data_analysis.dump_corpus(f"corpora/{data_analysis.name.replace(' ', '_')}.txt")
        figure = data_analysis.create_token_hist()
        print(f"Data set length: {len(data_analysis.data)}")

    print(f"Total unique tokens (across all datasets analyzed): {num_unique_tokens}")
    print(f"Total tokens: {num_tokens}")

    plt.show()
