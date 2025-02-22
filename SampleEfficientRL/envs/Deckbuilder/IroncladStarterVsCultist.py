from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
    DeckbuilderSingleBattleEnv,
)
from SampleEfficientRL.Envs.Deckbuilder.Decks.create_starter_ironclad_deck import (
    create_starter_ironclad_deck,
)
from SampleEfficientRL.Envs.Deckbuilder.Opponents.FixedHpCultist import FixedHpCultist
from SampleEfficientRL.Envs.Deckbuilder.Player import Player

IRONCLAD_STARTING_HP = 80
IRONCLAD_STARTING_ENERGY = 3


class IroncladStarterVsCultist(DeckbuilderSingleBattleEnv):
    def __init__(self) -> None:
        super().__init__()
        self.set_player(
            Player(
                self,
                create_starter_ironclad_deck(),
                IRONCLAD_STARTING_HP,
                IRONCLAD_STARTING_ENERGY,
            )
        )
        self.set_opponents([FixedHpCultist(self)])
