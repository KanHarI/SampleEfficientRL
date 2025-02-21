
from dataclasses import dataclass
from enum import Enum

class EnvActionType(Enum):
    ATTACK_ENEMY = "attack_enemy"

@dataclass
class EnvAction:
    env_action_type: EnvActionType
