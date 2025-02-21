from typing import Dict, Optional

from SampleEfficientRL.Envs.Deckbuilder.Card import (Card, CardEffectTrigger,
                                                     CardType, CardUIDs)
from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import \
    DeckbuilderSingleBattleEnv
from SampleEfficientRL.Envs.Deckbuilder.EffectCallback import EffectCallback
from SampleEfficientRL.Envs.Deckbuilder.Statuses.Block import Block


class Defend(Card):
    def __init__(self):
        super().__init__(card_type=CardType.SKILL, cost=1, card_uid=CardUIDs.DEFEND)

    def get_effects(self) -> Dict[CardEffectTrigger, EffectCallback]:
        return {CardEffectTrigger.ON_PLAY: Defend.on_play}

    @staticmethod
    def on_play(env: DeckbuilderSingleBattleEnv, _: Optional[int] = None) -> None:
        env.apply_effect_to_player(Block, 5)
