from typing import Dict, Optional

from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
    DeckbuilderSingleBattleEnv,
    EnvAction,
)
from SampleEfficientRL.Envs.Deckbuilder.Status import (
    EffectTriggerPoint,
    Status,
    StatusEffectCallback,
    StatusUIDs,
)


class HandDrawer(Status):
    def __init__(self) -> None:
        super().__init__(StatusUIDs.HAND_DRAWER)

    def get_effects(self) -> Dict[EffectTriggerPoint, StatusEffectCallback]:
        return {
            EffectTriggerPoint.ON_START_OF_TURN: self.on_start_of_turn,
            EffectTriggerPoint.ON_END_OF_TURN: self.on_end_of_turn,
        }

    @staticmethod
    def on_start_of_turn(
        env: DeckbuilderSingleBattleEnv, amount: int, action: EnvAction
    ) -> Optional[EnvAction]:
        if action.entity_descriptor.is_player is False:
            raise ValueError("HandDrawer status can only be applied to the player")
        for _ in range(amount):
            env.player_draw_card()
        return action

    @staticmethod
    def on_end_of_turn(
        env: DeckbuilderSingleBattleEnv, amount: int, action: EnvAction
    ) -> Optional[EnvAction]:
        if action.entity_descriptor.is_player is False:
            raise ValueError("HandDrawer status can only be applied to the player")
        env.player_discard_hand()
        return action
