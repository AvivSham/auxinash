from abc import abstractmethod, ABC
from torch import nn
from typing import Iterator


class AbstractMTLModel(ABC, nn.Module):
    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.shared.parameters()

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.task_specific.parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters of the last shared layer.
        Returns
        -------
        """
        return self.last_shared.parameters()

    @abstractmethod
    def forward(self, x, return_representation=False):
        pass
