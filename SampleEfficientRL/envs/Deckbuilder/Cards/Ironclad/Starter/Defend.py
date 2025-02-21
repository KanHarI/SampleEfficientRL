from typing import Dict, Optional

from SampleEfficientRL.Envs.Deckbuilder.Card import (
    Card,
    CardEffectCallback,
    CardEffectTrigger,
    CardType,
    CardUIDs,
)
from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
    DeckbuilderSingleBattleEnv,
    EntityDescriptor,
)
from SampleEfficientRL.Envs.Deckbuilder.Statuses.Block import Block

DEFEND_BLOCK_AMOUNT = 5


class Defend(Card):
    def __init__(self):
        super().__init__(card_type=CardType.SKILL, cost=1, card_uid=CardUIDs.DEFEND)

    def get_effects(self) -> Dict[CardEffectTrigger, CardEffectCallback]:
        return {CardEffectTrigger.ON_PLAY: Defend.on_play}

    @staticmethod
    def on_play(env: DeckbuilderSingleBattleEnv, _: Optional[int] = None) -> None:
        env.apply_status_to_entity(
            entity_descriptor=EntityDescriptor(is_player=True, enemy_idx=None),
            status=Block(),
            amount=DEFEND_BLOCK_AMOUNT,
        )
