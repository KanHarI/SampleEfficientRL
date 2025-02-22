import math
from typing import Dict, Optional, cast

from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
    DeckbuilderSingleBattleEnv,
    EnvAction,
    EnvActionType,
)
from SampleEfficientRL.Envs.Deckbuilder.EnvActions.Attack import Attack
from SampleEfficientRL.Envs.Deckbuilder.Status import (
    EffectTriggerPoint,
    Status,
    StatusEffectCallback,
    StatusUIDs,
)


class Vulnerable(Status):
    def __init__(self):
        super().__init__(StatusUIDs.VULNERABLE)

    def get_effects(self) -> Dict[EffectTriggerPoint, StatusEffectCallback]:
        return {
            EffectTriggerPoint.ON_ATTACKED: Vulnerable.on_attacked,
            EffectTriggerPoint.ON_END_OF_TURN: Vulnerable.on_end_of_turn,
        }

    @staticmethod
    def on_attacked(
        env: DeckbuilderSingleBattleEnv, amount: int, action: EnvAction
    ) -> Optional[EnvAction]:
        if action.env_action_type != EnvActionType.ATTACK:
            raise ValueError(
                f"Vulnerable status can only be applied to attack actions, not {action.env_action_type}"
            )

        attack_action = cast(Attack, action)
        attack_action.damage = math.floor(attack_action.damage * 1.5)
        return attack_action

    @staticmethod
    def on_end_of_turn(
        env: DeckbuilderSingleBattleEnv, amount: int, action: EnvAction
    ) -> Optional[EnvAction]:
        if amount == 1:
            env.reset_entity_status(action.entity_descriptor, StatusUIDs.VULNERABLE)
        else:
            env.apply_status_to_entity(action.entity_descriptor, Vulnerable(), -1)
        return action
