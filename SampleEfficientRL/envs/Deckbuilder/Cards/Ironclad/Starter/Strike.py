from typing import Dict, Optional

from SampleEfficientRL.Envs.Deckbuilder.Card import (Card, CardEffectCallback,
                                                     CardEffectTrigger,
                                                     CardType, CardUIDs)
from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import \
    DeckbuilderSingleBattleEnv


class Strike(Card):
    def __init__(self):
        super().__init__(card_type=CardType.ATTACK, cost=1, card_uid=CardUIDs.STRIKE)

    def get_effects(self) -> Dict[CardEffectTrigger, CardEffectCallback]:
        return {CardEffectTrigger.ON_PLAY: Strike.on_play}

    @staticmethod
    def on_play(
        env: DeckbuilderSingleBattleEnv, enemy_idx: Optional[int] = None
    ) -> None:
        if enemy_idx is None:
            raise ValueError("Enemy IDs are required for Strike")

        env.attack_enemy(enemy_idx, 6)
