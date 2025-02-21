
from dataclasses import dataclass
from typing import Optional
from SampleEfficientRL.Envs.Deckbuilder.EnvAction import EnvAction

@dataclass
class Attack(EnvAction):
    target_is_enemy: bool
    enemy_idx: Optional[int]
    damage: int
