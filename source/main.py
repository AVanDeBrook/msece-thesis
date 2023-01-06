from typing import *
from data import Data, TokenStats
from atccomplete import ATCCompleteData
from atcosim import ATCOSimData
from atco2sim import ATCO2SimData
from czechdataset import ZCUATCDataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # random seed initialization (passed to data classes) for reproducible randomness
    # in results
    RANDOM_SEED: int = 1

    # root dataset paths corresponding to data analysis classes
    datasets: Dict[str, Data] = {
        "~/data/atc0_comp_LDC94S14A/atc0_comp": ATCCompleteData,
        "~/data/ATCO": ATCOSimData,
        "~/data/ATCO2": ATCO2SimData,
        "~/data/ZCU_CZ_ATC": ZCUATCDataset,
    }

    for root_path, data_class in datasets.items():
        data_analysis: Data = data_class(data_root=root_path, random_seed=RANDOM_SEED)

        print(data_analysis.name)
        # parse transcripts in dataset
        data_analysis.parse_transcripts()
        data_analysis.dump_corpus(f"{data_analysis.name}.txt")
        # utterance stats
        stats = data_analysis.calc_token_stats()

        # utterance distribution
        utterance_hist = data_analysis.create_token_hist()
        utterance_hist.plot(pyplot=True)
        plt.show()
