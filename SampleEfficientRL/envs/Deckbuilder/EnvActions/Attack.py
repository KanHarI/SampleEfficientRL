from dataclasses import dataclass
from typing import Optional

from SampleEfficientRL.Envs.Deckbuilder.EnvAction import EnvAction


@dataclass
class Attack(EnvAction):
    damage: int
