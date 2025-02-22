from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
        DeckbuilderSingleBattleEnv,
    )

from SampleEfficientRL.Envs.Deckbuilder.EnvAction import EnvAction
from SampleEfficientRL.Envs.Deckbuilder.Status import (
    EffectTriggerPoint,
    Status,
    StatusEffectCallback,
    StatusUIDs,
)


class EnergyUser(Status):
    def __init__(self) -> None:
        super().__init__(StatusUIDs.ENERGY_USER)

    def get_effects(self) -> Dict[EffectTriggerPoint, StatusEffectCallback]:
        return {
            EffectTriggerPoint.ON_START_OF_TURN: EnergyUser.on_start_of_turn,
            EffectTriggerPoint.ON_END_OF_TURN: EnergyUser.on_end_of_turn,
        }

    @staticmethod
    def on_start_of_turn(
        env: "DeckbuilderSingleBattleEnv", amount: int, action: EnvAction
    ) -> Optional[EnvAction]:
        if action.entity_descriptor.is_player is False:
            raise ValueError("EnergyUser status can only be applied to the player")
        if env.player is None:
            raise ValueError("Player is not set")
        env.player.energy = amount
        return action

    @staticmethod
    def on_end_of_turn(
        env: "DeckbuilderSingleBattleEnv", amount: int, action: EnvAction
    ) -> Optional[EnvAction]:
        if action.entity_descriptor.is_player is False:
            raise ValueError("EnergyUser status can only be applied to the player")
        if env.player is None:
            raise ValueError("Player is not set")
        env.player.energy = 0
        return action
