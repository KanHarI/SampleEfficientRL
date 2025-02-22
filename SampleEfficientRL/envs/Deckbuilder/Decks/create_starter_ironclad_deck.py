from typing import List

from SampleEfficientRL.Envs.Deckbuilder.Card import Card
from SampleEfficientRL.Envs.Deckbuilder.Cards.Ironclad.Starter.Bash import Bash
from SampleEfficientRL.Envs.Deckbuilder.Cards.Ironclad.Starter.Defend import Defend
from SampleEfficientRL.Envs.Deckbuilder.Cards.Ironclad.Starter.Strike import Strike


def create_starter_ironclad_deck() -> List[Card]:
    cards: List[Card] = []
    cards.append(Bash())
    for _ in range(5):
        cards.append(Strike())
    for _ in range(4):
        cards.append(Defend())
    return cards
