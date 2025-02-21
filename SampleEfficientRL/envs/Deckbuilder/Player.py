from typing import List

from SampleEfficientRL.Envs.Deckbuilder.Card import Card
from SampleEfficientRL.Envs.Deckbuilder.Entity import Entity


class Player(Entity):
    def __init__(self, starting_deck: List[Card], max_health: int):
        self.deck = starting_deck
        super().__init__(max_health)
