from enum import Enum


class DataSplit(Enum):
    """
    Describes the different splits of the data.
    """
    TRAINING = 0
    TEST = 1
    VALIDATION = 2