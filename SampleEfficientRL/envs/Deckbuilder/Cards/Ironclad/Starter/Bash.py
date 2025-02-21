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
from SampleEfficientRL.Envs.Deckbuilder.Statuses.Vulnerable import Vulnerable

BASH_CARD_DAMAGE = 8
BASH_AMOUNT_OF_VULNERABLE = 2


class Bash(Card):
    def __init__(self):
        super().__init__(card_type=CardType.ATTACK, cost=2, card_uid=CardUIDs.BASH)

    def get_effects(self) -> Dict[CardEffectTrigger, CardEffectCallback]:
        return {CardEffectTrigger.ON_PLAY: Bash.on_play}

    @staticmethod
    def on_play(
        env: DeckbuilderSingleBattleEnv, enemy_idx: Optional[int] = None
    ) -> None:
        if enemy_idx is None:
            raise ValueError("Enemy IDs are required for Bash")
        num_opponents_before_attack = env.get_num_opponents()
        env.attack_entity(
            entity_descriptor=EntityDescriptor(is_player=False, enemy_idx=enemy_idx),
            amount=BASH_CARD_DAMAGE,
        )
        if num_opponents_before_attack == env.get_num_opponents():
            env.apply_status_to_entity(
                entity_descriptor=EntityDescriptor(
                    is_player=False, enemy_idx=enemy_idx
                ),
                status=Vulnerable(),
                amount=BASH_AMOUNT_OF_VULNERABLE,
            )
