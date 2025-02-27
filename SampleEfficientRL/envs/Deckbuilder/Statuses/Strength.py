from typing import TYPE_CHECKING, Dict, Optional, cast

if TYPE_CHECKING:
    from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
        DeckbuilderSingleBattleEnv,
    )

from SampleEfficientRL.Envs.Deckbuilder.EnvAction import EnvAction, EnvActionType
from SampleEfficientRL.Envs.Deckbuilder.EnvActions.Attack import Attack
from SampleEfficientRL.Envs.Deckbuilder.Status import (
    EffectTriggerPoint,
    Status,
    StatusEffectCallback,
    StatusUIDs,
)


class Strength(Status):
    def __init__(self) -> None:
        super().__init__(StatusUIDs.STRENGTH)

    def get_effects(self) -> Dict[EffectTriggerPoint, StatusEffectCallback]:
        return {EffectTriggerPoint.ON_ATTACK: Strength.on_attack}

    @staticmethod
    def on_attack(
        env: "DeckbuilderSingleBattleEnv", amount: int, action: EnvAction
    ) -> Optional[EnvAction]:
        if action.env_action_type != EnvActionType.ATTACK:
            raise ValueError(
                f"Strength status can only be applied to attack actions, not {action.env_action_type}"
            )
        attack_action = cast(Attack, action)
        attack_action.damage += amount
        return attack_action
