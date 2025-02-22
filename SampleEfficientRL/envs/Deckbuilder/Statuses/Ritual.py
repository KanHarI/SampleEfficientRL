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
from SampleEfficientRL.Envs.Deckbuilder.Statuses.Strength import Strength


class Ritual(Status):
    def __init__(self):
        super().__init__(StatusUIDs.RITUAL)

    def get_effects(self) -> Dict[EffectTriggerPoint, StatusEffectCallback]:
        return {EffectTriggerPoint.ON_START_OF_TURN: Ritual.on_start_of_turn}

    @staticmethod
    def on_start_of_turn(
        env: DeckbuilderSingleBattleEnv, amount: int, action: EnvAction
    ) -> Optional[EnvAction]:
        env.apply_status_to_entity(action.entity_descriptor, Strength(), amount)
        return action
