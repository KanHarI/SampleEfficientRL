from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Dict, List, Optional

from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import \
    DeckbuilderSingleBattleEnv
from SampleEfficientRL.Envs.Deckbuilder.EnvAction import EnvAction


class EffectTriggerPoint(Enum):
    ON_ATTACKED = 1
    ON_DEFEND = 2
    ON_END_OF_TURN = 3
    ON_ATTACK = 4
    ON_START_OF_TURN = 5
    ON_DEATH = 6


class StatusUIDs(Enum):
    VULNERABLE = 1
    WEAK = 2
    FRAIL = 3
    POISON = 4
    BLOCK = 5


StatusesOrder: List[StatusUIDs] = [
    StatusUIDs.VULNERABLE,
    StatusUIDs.WEAK,
    StatusUIDs.FRAIL,
    StatusUIDs.BLOCK,
    StatusUIDs.POISON,
]

# env, amount, action
StatusEffectCallback = Callable[
    [DeckbuilderSingleBattleEnv, int, EnvAction], Optional[EnvAction]
]


class Status(ABC):
    def __init__(self, status_uid: StatusUIDs):
        self.status_uid = status_uid

    @abstractmethod
    def get_effects(self) -> Dict[EffectTriggerPoint, EffectCallback]:
        pass
