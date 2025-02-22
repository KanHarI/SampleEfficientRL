from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
    DeckbuilderSingleBattleEnv,
)
from SampleEfficientRL.Envs.Deckbuilder.Entity import Entity


class NextMoveType(Enum):
    ATTACK = 0
    RITUAL = 1


class NextMove(Enum):
    def __init__(self, move_type: NextMoveType, amount: Optional[int] = None):
        self.move_type = move_type
        self.amount = amount


class Opponent(ABC, Entity):
    def __init__(self, env: DeckbuilderSingleBattleEnv, max_health: int):
        super().__init__(env, max_health)

    @abstractmethod
    def select_move(self) -> NextMove:
        pass
