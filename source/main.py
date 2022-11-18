from typing import *
from hiwire import HIWIREData
from data import Data, UtteranceStats
from atccomplete import ATCCompleteData
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # random seed initialization (passed to data classes) for reproducible randomness
    # in results
    RANDOM_SEED: int = 1

    # root dataset paths corresponding to data analysis classes
    datasets: Dict[str, Data]  = {
        "/data/s0293/S0293/speechdata": HIWIREData,
        "/data/atc0_comp_ldc945s14a/atc0_comp_raw": ATCCompleteData,
    }

    for root_path, data_class in datasets.items():
        data_analysis: Data = data_class(
            data_root=root_path, random_seed=RANDOM_SEED
        )

        print(data_analysis.name)
        # parse transcripts in dataset
        data_analysis.parse_transcripts()
        data_analysis.dump_manifest(f"{data_analysis.name}_all.json")
        # utterance stats
        # stats = data_analysis.calc_utterance_stats()

        # utterance distribution
        # utterance_hist = data_analysis.create_utterance_hist()
        # utterance_hist.plot(pyplot=True).show()

        # spectrogram plots, randomly sampled (depending on the size of the dataset,
        # this can take up a lot of time and memory)
        spec = data_analysis.create_spectrograms(random_sample=False)
        # reasoning for using plt.show() instead of spec.show() from matplotlib docs:
        # "If you're running a pure python shell or executing a non-GUI python script,
        #  you should use matplotlib.pyplot.show instead, which takes care of managing
        #  the event loop for you."
        plt.show()
        # # spectrogram plots, in order
        # data_analysis.create_spectrograms(random_sample=False)
