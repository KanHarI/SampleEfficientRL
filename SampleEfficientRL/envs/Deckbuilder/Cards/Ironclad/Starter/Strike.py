
from typing import Dict, Optional
from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import DeckbuilderSingleBattleEnv
from SampleEfficientRL.Envs.Deckbuilder.EffectCallback import EffectCallback
from SampleEfficientRL.Envs.Deckbuilder.Card import CardEffectTrigger, CardType, CardUIDs


class Strike(Card):
    def __init__(self):
        super().__init__(
            card_type=CardType.ATTACK,
            cost=1,
            card_uid=CardUIDs.STRIKE
        )

    def get_effects(self) -> Dict[CardEffectTrigger, EffectCallback]:
        return {CardEffectTrigger.ON_PLAY: Strike.on_play}
    
    @staticmethod
    def on_play(
        env: DeckbuilderSingleBattleEnv, enemy_idx: Optional[int] = None
    ) -> None:
        if enemy_idx is None:
            raise ValueError("Enemy IDs are required for Strike")
        
        env.deal_damage_to_enemy(enemy_idx, 6)
