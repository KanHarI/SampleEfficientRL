from typing import List

from SampleEfficientRL.Envs.Deckbuilder.Opponent import Opponent
from SampleEfficientRL.Envs.Deckbuilder.Player import Player
from SampleEfficientRL.Envs.Env import Env


class DeckbuilderSingleBattleEnv(Env):
    def __init__(self, player: Player, opponents: List[Opponent]):
        self.player = player
        self.opponents = opponents
