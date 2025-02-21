
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class EnvActionType(Enum):
    ATTACK = "attack"


@dataclass
class EnvAction:
    env_action_type: EnvActionType
    action_on_player: bool
    action_on_enemy: bool
    enemy_idx: Optional[int]
    active_amount: Optional[int]
