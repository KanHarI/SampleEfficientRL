from dataclasses import dataclass

from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import EnvAction


@dataclass
class StartOfTurn(EnvAction):
    pass
