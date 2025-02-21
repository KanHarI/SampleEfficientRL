from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CardType(Enum):
    POWER = 1
    SKILL = 2
    ATTACK = 3
    CURSE = 4
    STATUS = 5


class TargetType(Enum):
    OPPONENT = 1
    CARD = 2


@dataclass
class CardTargetingInfo:
    requires_target: bool
    targeting_type: Optional[TargetType]


class Card(ABC):
    def __init__(self, name: str, card_type: CardType, cost: int):
        self.name = name
        self.card_type = card_type
        self.cost = cost

    @abstractmethod
    def effect(self):
        pass

    @abstractmethod
    def upgrade(self) -> Card:
        pass

    @abstractmethod
    def get_targeting_info(self) -> CardTargetingInfo:
        pass
