import random
from typing import TYPE_CHECKING, List, Optional

from SampleEfficientRL.Envs.Deckbuilder.Card import Card
from SampleEfficientRL.Envs.Deckbuilder.Statuses.EnergyUser import EnergyUser

if TYPE_CHECKING:
    from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
        DeckbuilderSingleBattleEnv,
    )
from SampleEfficientRL.Envs.Deckbuilder.Entity import Entity
from SampleEfficientRL.Envs.Deckbuilder.EnvAction import EntityDescriptor
from SampleEfficientRL.Envs.Deckbuilder.Statuses.HandDrawer import HandDrawer

HAND_SIZE = 5


class Player(Entity):
    hand: List[Card] = []
    draw_pile: List[Card] = []
    discard_pile: List[Card]

    def __init__(
        self,
        env: "DeckbuilderSingleBattleEnv",
        starting_deck: List[Card],
        max_health: int,
        max_energy: int,
    ):
        self.deck = starting_deck
        self.discard_pile = starting_deck
        self.max_energy = max_energy
        self.energy = 0
        super().__init__(env, max_health)

    def register_player(self, env: "DeckbuilderSingleBattleEnv") -> None:
        env.apply_status_to_entity(
            EntityDescriptor(is_player=True), HandDrawer(), HAND_SIZE
        )
        env.apply_status_to_entity(
            EntityDescriptor(is_player=True), EnergyUser(), self.max_energy
        )

    def draw_card(self) -> None:
        if len(self.draw_pile) == 0:
            self.draw_pile = self.discard_pile
            self.discard_pile = []
            random.shuffle(self.draw_pile)
        if len(self.draw_pile) > 0:
            self.hand.append(self.draw_pile.pop())

    def discard_hand(self) -> None:
        self.discard_pile.extend(self.hand)
        self.hand = []

    def play_card(self, card_idx: int, target_idx: Optional[int] = None) -> None:
        if card_idx < 0 or card_idx >= len(self.hand):
            raise ValueError(f"Invalid card index: {card_idx}")
        card = self.hand[card_idx]
        self.energy -= card.cost
        self.hand.pop(card_idx)
        self.env.play_card(card, target_idx)
        self.discard_pile.append(card)
