from data import Data


class ATCO2SimData(Data):
    """
    This class defines the format for the Air Traffic Control 2 dataset and implements
    functions to parse the data into a common format for data analysis.

    This dataset is described in more depth and can be obtained here:
        https://www.atco2.org/data
    """

    def __init__(self, data_root: str):
        super(ATCO2SimData, self).__init__(data_root)

    def name(self) -> str:
        return "ATCO2"
