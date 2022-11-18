from data import Data

class ATCCommunication(Data):
    """
    This class defines the format of the data for the Air Traffic Control Communication
    and implements functions to parse the data into a common format for data analysis.

    This dataset can be obtained from:
        https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCA1-0
    """

    def __init__(self, data_root: str):
        super(ATCCommunication, self).__init__(data_root)

    def name(self) -> str:
        return "ATCCommunication"
