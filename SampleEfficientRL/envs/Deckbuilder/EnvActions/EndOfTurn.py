from dataclasses import dataclass

from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import EnvAction


@dataclass
class EndOfTurn(EnvAction):
    pass
