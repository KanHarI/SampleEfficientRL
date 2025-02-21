
from typing import Dict, Optional, cast
from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import DeckbuilderSingleBattleEnv
from SampleEfficientRL.Envs.Deckbuilder.EffectCallback import EffectCallback
from SampleEfficientRL.Envs.Deckbuilder.EnvAction import EnvAction, EnvActionType
from SampleEfficientRL.Envs.Deckbuilder.EnvActions.Attack import Attack
from SampleEfficientRL.Envs.Deckbuilder.Status import EffectTriggerPoint, Status, StatusUIDs


class Block(Status):
    def __init__(self):
        super().__init__(StatusUIDs.BLOCK)

    def get_effects(self) -> Dict[EffectTriggerPoint, EffectCallback]:
        return {EffectTriggerPoint.ON_ATTACKED: self.on_attacked}
    
    def on_attacked(self, env: DeckbuilderSingleBattleEnv, action: EnvAction) -> Optional[EnvAction]:
        if action.env_action_type != EnvActionType.ATTACK:
            raise ValueError(f"Block status can only be applied to attack actions, not {action.env_action_type}")
        
        attack_action = cast(Attack, action)
        active_amount = self.get_active_amount()
        if attack_action.damage > active_amount:
            env.reset_status_from_action(action, StatusUIDs.BLOCK)
            attack_action.damage = attack_action.damage - active_amount
        else:
            env.apply_status_to_player(StatusUIDs.BLOCK, -attack_action.damage)
            return None
        