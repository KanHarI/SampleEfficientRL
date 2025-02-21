
from typing import Any, Dict
from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import DeckbuilderSingleBattleEnv
from SampleEfficientRL.Envs.Deckbuilder.EffectCallback import EffectCallback
from SampleEfficientRL.Envs.Deckbuilder.EnvAction import EnvAction, EnvActionType
from SampleEfficientRL.Envs.Deckbuilder.EnvActions.Attack import Attack
from SampleEfficientRL.Envs.Deckbuilder.Status import EffectTriggerPoint, Status, StatusUIDs

import math

class Vulnerable(Status):
    def __init__(self):
        super().__init__(StatusUIDs.VULNERABLE)

    def get_effects(self) -> Dict[EffectTriggerPoint, EffectCallback]:
        return {EffectTriggerPoint.ON_ATTACKED: self.on_attacked}

    def on_attacked(self, env: DeckbuilderSingleBattleEnv, action: EnvAction) -> EnvAction:
        if action.env_action_type != EnvActionType.ATTACK:
            raise ValueError(f"Vulnerable status can only be applied to attack actions, not {action.env_action_type}")
        
        return Attack(action.enemy_idx, math.floor(action.damage * 1.5))
