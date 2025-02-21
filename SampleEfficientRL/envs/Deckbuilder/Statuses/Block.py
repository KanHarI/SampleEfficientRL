from typing import Dict, Optional, cast

from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import \
    DeckbuilderSingleBattleEnv
from SampleEfficientRL.Envs.Deckbuilder.EnvAction import (EnvAction,
                                                          EnvActionType)
from SampleEfficientRL.Envs.Deckbuilder.EnvActions.Attack import Attack
from SampleEfficientRL.Envs.Deckbuilder.Status import (EffectTriggerPoint,
                                                       Status,
                                                       StatusEffectCallback,
                                                       StatusUIDs)


class Block(Status):
    def __init__(self):
        super().__init__(StatusUIDs.BLOCK)

    def get_effects(self) -> Dict[EffectTriggerPoint, StatusEffectCallback]:
        return {EffectTriggerPoint.ON_ATTACKED: self.on_attacked}

    def on_attacked(
        self, env: DeckbuilderSingleBattleEnv, amount: int, action: EnvAction
    ) -> Optional[EnvAction]:
        if action.env_action_type != EnvActionType.ATTACK:
            raise ValueError(
                f"Block status can only be applied to attack actions, not {action.env_action_type}"
            )

        attack_action = cast(Attack, action)
        if attack_action.damage > amount:
            env.reset_status_from_action(action, StatusUIDs.BLOCK)
            attack_action.damage = attack_action.damage - amount
            return attack_action
        else:
            env.apply_status_to_player(StatusUIDs.BLOCK, -attack_action.damage)
            return None
