from abc import ABC

from torch import nn


class BaseModule(nn.Module, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__()
