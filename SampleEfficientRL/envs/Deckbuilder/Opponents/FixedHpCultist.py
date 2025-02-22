from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
    DeckbuilderSingleBattleEnv,
)
from SampleEfficientRL.Envs.Deckbuilder.Opponents.Cultist import Cultist

FIXED_HP_CULTIST_MAX_HEALTH = 45


class FixedHpCultist(Cultist):
    def __init__(self, env: DeckbuilderSingleBattleEnv):
        super().__init__(env, FIXED_HP_CULTIST_MAX_HEALTH)
