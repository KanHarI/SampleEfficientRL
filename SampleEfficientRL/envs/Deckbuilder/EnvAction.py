from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass
class EntityDescriptor:
    is_player: bool
    enemy_idx: Optional[int] = None


class EnvActionType(Enum):
    ATTACK = "attack"
    START_OF_TURN = "start_of_turn"
    END_OF_TURN = "end_of_turn"


@dataclass
class EnvAction:
    env_action_type: EnvActionType
    entity_descriptor: EntityDescriptor
