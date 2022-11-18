from data import Data


class ATCOSimData(Data):
    """
    This class describes the format of the Air Traffic Control Simulation Corpus
    and defines functions for parsing the data into a common format for data analysis.

    This dataset is described in more depth and can be obtained here:
        https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html
    """

    def __init__(self, data_root: str, **kwargs):
        super(ATCOSimData, self).__init__(data_root, **kwargs)

    @property
    def name(self):
        return "ATCO"
