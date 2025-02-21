from typing import Dict, Optional

from SampleEfficientRL.Envs.Deckbuilder.Card import (Card, CardEffectTrigger,
                                                     CardType, CardUIDs)
from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import \
    DeckbuilderSingleBattleEnv
from SampleEfficientRL.Envs.Deckbuilder.EffectCallback import EffectCallback
from SampleEfficientRL.Envs.Deckbuilder.Statuses.Vulnerable import Vulnerable

BASH_CARD_DAMAGE = 8
BASH_AMOUNT_OF_VULNERABLE = 2


class Bash(Card):
    def __init__(self):
        super().__init__(card_type=CardType.ATTACK, cost=2, card_uid=CardUIDs.BASH)

    def get_effects(self) -> Dict[CardEffectTrigger, EffectCallback]:
        return {CardEffectTrigger.ON_PLAY: Bash.on_play}

    @staticmethod
    def on_play(
        env: DeckbuilderSingleBattleEnv, enemy_idx: Optional[int] = None
    ) -> None:
        if enemy_idx is None:
            raise ValueError("Enemy IDs are required for Bash")

        env.deal_damage_to_enemy(enemy_idx, BASH_CARD_DAMAGE)
        env.apply_status_to_enemy(enemy_idx, Vulnerable, BASH_AMOUNT_OF_VULNERABLE)
