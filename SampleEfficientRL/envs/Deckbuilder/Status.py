from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict
from dataclasses import dataclass

from SampleEfficientRL.Envs.Deckbuilder.EffectCallback import EffectCallback

class EffectTriggerPoint(Enum):
    ON_ATTACKED = 1
    ON_DEFEND = 2
    ON_END_OF_TURN = 3
    ON_ATTACK = 4
    ON_START_OF_TURN = 5


class StatusUIDs(Enum):
    VULNERABLE = 1
    WEAK = 2
    FRAIL = 3
    POISON = 4


class Status(ABC):
    def __init__(self, status_uid: StatusUIDs):
        self.status_uid = status_uid

    @abstractmethod
    def get_effects(self) -> Dict[EffectTriggerPoint, EffectCallback]:
        pass
