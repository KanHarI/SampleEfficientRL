from enum import Enum


class NextMove(Enum):
    ATTACK = 0
    BUFF = 1
    DEBUFF = 2


class Opponent:
    def __init__(self, max_health: int):
        self.max_health = max_health
        self.current_health = max_health
