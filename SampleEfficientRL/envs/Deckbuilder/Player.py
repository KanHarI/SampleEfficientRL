from typing import Dict, List, Tuple

from SampleEfficientRL.Envs.Deckbuilder.Card import Card
from SampleEfficientRL.Envs.Deckbuilder.Entity import Entity
from SampleEfficientRL.Envs.Deckbuilder.Status import Status, StatusUIDs


class Player(Entity):
    def __init__(self, starting_deck: List[Card], max_health: int):
        self.deck = starting_deck
        super().__init__(max_health)
