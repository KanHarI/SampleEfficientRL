from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from SampleEfficientRL.Envs.Deckbuilder.Entity import Entity
from SampleEfficientRL.Envs.Deckbuilder.EnvActions.Attack import Attack
from SampleEfficientRL.Envs.Deckbuilder.Opponent import Opponent
from SampleEfficientRL.Envs.Deckbuilder.Player import Player
from SampleEfficientRL.Envs.Deckbuilder.Status import (
    EffectTriggerPoint,
    Status,
    StatusesOrder,
    StatusUIDs,
)
from SampleEfficientRL.Envs.Env import Env


@dataclass
class EntityDescriptor:
    is_player: bool
    enemy_idx: Optional[int] = None


class EnvActionType(Enum):
    ATTACK = "attack"


@dataclass
class EnvAction:
    env_action_type: EnvActionType
    entity_descriptor: EntityDescriptor


class EnvEvents(Enum):
    PLAYER_DEATH = "PLAYER_DEATH"
    OPPONENT_DEATH = "OPPONENT_DEATH"
    WIN_BATTLE = "WIN_BATTLE"


class DeckbuilderSingleBattleEnv(Env):
    def __init__(self, player: Player, opponents: List[Opponent]):
        self.player = player
        self.opponents = opponents
        self.unemitted_events: List[EnvEvents] = []

    def find_entity_by_descriptor(self, entity_descriptor: EntityDescriptor) -> Entity:
        if entity_descriptor.is_player:
            return self.player
        else:
            if entity_descriptor.enemy_idx is None:
                raise ValueError("Enemy index is required for non-player entities")
            return self.opponents[entity_descriptor.enemy_idx]

    def get_num_opponents(self) -> int:
        return len(self.opponents)

    def reduce_entity_hp(self, entity_descriptor: EntityDescriptor, amount: int):
        entity = self.find_entity_by_descriptor(entity_descriptor)
        is_dead = entity.reduce_health(amount)
        if is_dead:
            if entity_descriptor.is_player:
                self.unemitted_events.append(EnvEvents.PLAYER_DEATH)
            else:
                if entity_descriptor.enemy_idx is None:
                    raise ValueError("Enemy index is required for non-player entities")
                self.unemitted_events.append(EnvEvents.OPPONENT_DEATH)
                self.opponents.pop(entity_descriptor.enemy_idx)
                if len(self.opponents) == 0:
                    self.unemitted_events.append(EnvEvents.WIN_BATTLE)

    def emit_events(self) -> List[EnvEvents]:
        events = self.unemitted_events
        self.unemitted_events = []
        return events

    def apply_action_callbacks(
        self,
        action: EnvAction,
        entity_descriptor: EntityDescriptor,
        trigger_point: EffectTriggerPoint,
    ):
        entity = self.find_entity_by_descriptor(entity_descriptor)
        entity_active_statuses = entity.get_active_statuses()
        for status_sid in StatusesOrder:
            if status_sid in entity_active_statuses:
                status, amount = entity_active_statuses[status_sid]
                callback = status.get_effects()[trigger_point]
                if callback is not None:
                    action_or_none = callback(self, amount, action)
                if action_or_none is None:
                    break  # Action was fully blocked, etc.
                else:
                    action = action_or_none
        return action

    def attack_entity(self, entity_descriptor: EntityDescriptor, amount: int):
        action = Attack(
            env_action_type=EnvActionType.ATTACK,
            entity_descriptor=entity_descriptor,
            damage=amount,
        )
        action = self.apply_action_callbacks(
            action, entity_descriptor, EffectTriggerPoint.ON_ATTACKED
        )

        if action is not None:
            self.reduce_entity_hp(entity_descriptor, action.damage)

    def reset_entity_status(
        self, entity_descriptor: EntityDescriptor, status_uid: StatusUIDs
    ):
        entity = self.find_entity_by_descriptor(entity_descriptor)
        entity.reset_status(status_uid)

    def apply_status_to_entity(
        self, entity_descriptor: EntityDescriptor, status: Status, amount: int
    ):
        entity = self.find_entity_by_descriptor(entity_descriptor)
        entity.apply_status(status, amount)
