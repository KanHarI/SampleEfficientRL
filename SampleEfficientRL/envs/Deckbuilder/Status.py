from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict

from SampleEfficientRL.Envs.Deckbuilder.EffectCallback import EffectCallback


class EffectTriggerPoint(Enum):
    ON_ATTACKED = 1
    ON_DEFEND = 2
    ON_END_OF_TURN = 3
    ON_ATTACK = 4


class StatusUIDs(Enum):
    VULNERABLE = 1
    WEAK = 2
    FRAIL = 3
    POISON = 4


class Effect(ABC):
    def __init__(self, effect_uid: StatusUIDs, amount: int):
        self.effect_uid = effect_uid
        self.amount = amount

    @abstractmethod
    def get_effects(self) -> Dict[EffectTriggerPoint, EffectCallback]:
        pass
