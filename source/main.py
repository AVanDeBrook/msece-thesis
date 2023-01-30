from typing import *
from pprint import pprint
from data import Data, ATCCompleteData, ATCOSimData, ATCO2SimData, ZCUATCDataset
from matplotlib.figure import Figure
from seaborn.objects import Plot
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # random seed initialization (passed to data classes) for reproducible randomness
    # in results
    RANDOM_SEED: int = 1

    # root dataset paths corresponding to data analysis classes
    datasets: Dict[str, Data] = {
        # TODO: find a way to sync file paths across computers (shell/env var, config file?)
        "/home/avandebrook/thesis_data/atc0_comp": ATCCompleteData,
        "/home/avandebrook/thesis_data/atcosim/": ATCOSimData,
        "/home/avandebrook/thesis_data/ATCO2-ASRdataset-v1_beta": ATCO2SimData,
        "/home/avandebrook/thesis_data/ZCU_CZ_ATC": ZCUATCDataset,
    }

    # find number of unique tokens across datasets
    num_unique_tokens = 0
    num_tokens = 0

    for root_path, data_class in datasets.items():
        data_analysis: Data = data_class(data_root=root_path, random_seed=RANDOM_SEED)
        print(data_analysis.name)

        # parse transcripts in dataset
        data_analysis.parse_transcripts()
        token_freq = data_analysis.token_freq_analysis(normalize=True)
        num_unique_tokens += len(token_freq.keys())

        for counts in token_freq.values():
            num_tokens += counts[0]

        pprint(token_freq)
        # data_analysis.dump_corpus(f"{data_analysis.name}.txt")
        figure = data_analysis.create_token_hist()
        # matplotlib show if matplotlib, otherwise use show method from seaborn plot
        plt.show() if isinstance(figure, Figure) else figure.show()
        print("Done")

    print(f"Total unique tokens (across all datasets analyzed: {num_unique_tokens}")
    print(f"Total tokens: {num_tokens}")
