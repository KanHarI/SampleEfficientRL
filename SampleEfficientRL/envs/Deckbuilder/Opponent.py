from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
        DeckbuilderSingleBattleEnv,
    )

from SampleEfficientRL.Envs.Deckbuilder.Entity import Entity
from SampleEfficientRL.Envs.Deckbuilder.EnvAction import EntityDescriptor
from SampleEfficientRL.Envs.Deckbuilder.Statuses.Ritual import Ritual


class NextMoveType(Enum):
    ATTACK = 1
    RITUAL = 2


class OpponentTypeUIDs(Enum):
    CULTIST = 1


class NextMove(Enum):
    def __init__(self, move_type: NextMoveType, amount: Optional[int] = None):
        self.move_type = move_type
        self.amount = amount


class Opponent(ABC, Entity):
    def __init__(
        self,
        env: "DeckbuilderSingleBattleEnv",
        opponent_type_uid: OpponentTypeUIDs,
        max_health: int,
    ):
        self.opponent_type_uid = opponent_type_uid
        super().__init__(env, max_health)

    @abstractmethod
    def select_move(self, num_turn: int) -> NextMove:
        pass

    def perform_move(self, me: EntityDescriptor, move: NextMove) -> None:
        if move.move_type == NextMoveType.ATTACK:
            if move.amount is None:
                raise ValueError("Attack move amount cannot be None")
            self.env.attack_entity(EntityDescriptor(is_player=True), move.amount)
        elif move.move_type == NextMoveType.RITUAL:
            if move.amount is None:
                raise ValueError("Ritual move amount cannot be None")
            self.env.apply_status_to_entity(me, Ritual(), move.amount)
        else:
            raise NotImplementedError(f"Move type {move.move_type} not implemented")
