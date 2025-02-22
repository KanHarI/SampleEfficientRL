from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TAction = TypeVar("TAction")
TObservation = TypeVar("TObservation")


class Env(ABC, Generic[TAction, TObservation]):
    @abstractmethod
    def reset(self) -> TObservation:
        pass

    @abstractmethod
    def step(self, action: TAction) -> None:
        pass

    @abstractmethod
    def observe(self) -> TObservation:
        pass
