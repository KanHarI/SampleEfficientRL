from enum import Enum
from typing import Dict, Tuple

from SampleEfficientRL.Envs.Deckbuilder.Status import Status, StatusUIDs


class NextMove(Enum):
    ATTACK = 0
    BUFF = 1
    DEBUFF = 2


class Opponent:
    statuses: Dict[StatusUIDs, Tuple[Status, int]]

    def __init__(self, max_health: int):
        self.max_health = max_health
        self.current_health = max_health
        self.statuses = {}
