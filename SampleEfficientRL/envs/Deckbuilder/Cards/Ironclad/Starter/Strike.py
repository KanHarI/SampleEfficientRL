from typing import TYPE_CHECKING, Dict, Optional

from SampleEfficientRL.Envs.Deckbuilder.Card import (
    Card,
    CardEffectCallback,
    CardEffectTrigger,
    CardTargetingInfo,
    CardType,
    CardUIDs,
    TargetType,
)

if TYPE_CHECKING:
    from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
        DeckbuilderSingleBattleEnv,
    )

from SampleEfficientRL.Envs.Deckbuilder.EnvAction import EntityDescriptor

STRIKE_DAMAGE = 6


class Strike(Card):
    def __init__(self) -> None:
        super().__init__(card_type=CardType.ATTACK, cost=1, card_uid=CardUIDs.STRIKE)

    def get_effects(self) -> Dict[CardEffectTrigger, CardEffectCallback]:
        return {CardEffectTrigger.ON_PLAY: Strike.on_play}

    @staticmethod
    def on_play(
        env: "DeckbuilderSingleBattleEnv", enemy_idx: Optional[int] = None
    ) -> None:
        if enemy_idx is None:
            raise ValueError("Enemy IDs are required for Strike")

        env.attack_entity(
            source=EntityDescriptor(is_player=True),
            target=EntityDescriptor(is_player=False, enemy_idx=enemy_idx),
            amount=STRIKE_DAMAGE,
        )

    def get_targeting_info(self) -> CardTargetingInfo:
        return CardTargetingInfo(
            requires_target=True, targeting_type=TargetType.OPPONENT
        )
