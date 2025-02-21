from typing import List

from SampleEfficientRL.Envs.Deckbuilder.Card import Card


class Player:
    def __init__(self, starting_deck: List[Card], max_health: int):
        self.deck = starting_deck
        self.max_health = max_health
        self.current_health = max_health
