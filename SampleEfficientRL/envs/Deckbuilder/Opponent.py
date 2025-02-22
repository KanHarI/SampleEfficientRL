from abc import ABC, abstractmethod
from dataclasses import dataclass
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


@dataclass
class NextMove:
    move_type: NextMoveType
    amount: Optional[int] = None


class Opponent(ABC, Entity):
    next_move: Optional[NextMove] = None

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
            self.env.attack_entity(
                source=me, target=EntityDescriptor(is_player=True), amount=move.amount
            )
        elif move.move_type == NextMoveType.RITUAL:
            if move.amount is None:
                raise ValueError("Ritual move amount cannot be None")
            self.env.apply_status_to_entity(me, Ritual(), move.amount)
        else:
            raise NotImplementedError(f"Move type {move.move_type} not implemented")

    def start_turn(self) -> None:
        self.next_move = self.select_move(self.env.num_turn)

    def perform_next_move(self, me: EntityDescriptor) -> None:
        if self.next_move is None:
            raise ValueError("Next move is not set")
        self.perform_move(me, self.next_move)
