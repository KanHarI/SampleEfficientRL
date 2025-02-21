from dataclasses import dataclass
from typing import List, Optional

from SampleEfficientRL.Envs.Deckbuilder.EnvActions.Attack import Attack
from SampleEfficientRL.Envs.Deckbuilder.Opponent import Opponent
from SampleEfficientRL.Envs.Deckbuilder.Player import Player
from SampleEfficientRL.Envs.Env import Env


@dataclass
class EntityDescriptor:
    is_player: bool
    enemy_idx: Optional[int]


class DeckbuilderSingleBattleEnv(Env):
    def __init__(self, player: Player, opponents: List[Opponent]):
        self.player = player
        self.opponents = opponents

    def reduce_entity_hp(self, entity_descriptor: EntityDescriptor, amount: int):
        if entity_descriptor.is_player:
            self.player.reduce_hp(amount)
        else:
            self.opponents[entity_descriptor.enemy_idx].reduce_hp(amount)

    def attack_entity(self, entity_descriptor: EntityDescriptor, amount: int):
        action = Attack(
            env_action_type=EnvActionType.ATTACK,
            action_on_player=entity_descriptor.is_player,
            action_on_enemy=not entity_descriptor.is_player,
            enemy_idx=entity_descriptor.enemy_idx,
            damage=amount,
        )
