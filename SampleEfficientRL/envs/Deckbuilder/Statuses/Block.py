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


class Block(Status):
    def __init__(self) -> None:
        super().__init__(StatusUIDs.BLOCK)

    def get_effects(self) -> Dict[EffectTriggerPoint, StatusEffectCallback]:
        return {
            EffectTriggerPoint.ON_ATTACKED: Block.on_attacked,
            EffectTriggerPoint.ON_END_OF_TURN: Block.on_end_of_turn,
        }

    @staticmethod
    def on_attacked(
        env: DeckbuilderSingleBattleEnv, amount: int, action: EnvAction
    ) -> Optional[EnvAction]:
        if action.env_action_type != EnvActionType.ATTACK:
            raise ValueError(
                f"Block status can only be applied to attack actions, not {action.env_action_type}"
            )

        attack_action = cast(Attack, action)
        if attack_action.damage > amount:
            env.reset_entity_status(action.entity_descriptor, StatusUIDs.BLOCK)
            attack_action.damage = attack_action.damage - amount
            return attack_action
        else:
            env.apply_status_to_entity(
                action.entity_descriptor, Block(), -attack_action.damage
            )
            return None

    @staticmethod
    def on_end_of_turn(
        env: DeckbuilderSingleBattleEnv, amount: int, action: EnvAction
    ) -> Optional[EnvAction]:
        env.reset_entity_status(action.entity_descriptor, StatusUIDs.BLOCK)
        return action
