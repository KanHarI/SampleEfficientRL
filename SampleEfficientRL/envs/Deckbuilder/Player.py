from typing import List

from SampleEfficientRL.Envs.Deckbuilder.Card import Card
from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
    DeckbuilderSingleBattleEnv,
)
from SampleEfficientRL.Envs.Deckbuilder.Entity import Entity


class Player(Entity):
    def __init__(
        self,
        env: DeckbuilderSingleBattleEnv,
        starting_deck: List[Card],
        max_health: int,
        max_energy: int,
    ):
        self.deck = starting_deck
        self.max_energy = max_energy
        super().__init__(env, max_health)
