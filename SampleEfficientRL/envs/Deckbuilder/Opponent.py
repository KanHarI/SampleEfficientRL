from enum import Enum
from typing import Dict, Tuple

from SampleEfficientRL.Envs.Deckbuilder.Entity import Entity
from SampleEfficientRL.Envs.Deckbuilder.Status import Status, StatusUIDs


class NextMove(Enum):
    ATTACK = 0
    BUFF = 1
    DEBUFF = 2


class Opponent(Entity):
    def __init__(self, max_health: int):
        super().__init__(max_health)
