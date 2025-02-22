from dataclasses import dataclass

from SampleEfficientRL.Envs.Deckbuilder.EnvAction import EnvAction


@dataclass
class Attack(EnvAction):
    damage: int
