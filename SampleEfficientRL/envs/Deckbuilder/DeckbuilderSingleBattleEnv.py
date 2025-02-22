from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, cast

from SampleEfficientRL.Envs.Deckbuilder.Entity import Entity
from SampleEfficientRL.Envs.Deckbuilder.EnvActions.Attack import Attack
from SampleEfficientRL.Envs.Deckbuilder.EnvActions.EndOfTurn import EndOfTurn
from SampleEfficientRL.Envs.Deckbuilder.EnvActions.StartOfTurn import StartOfTurn
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
    START_OF_TURN = "start_of_turn"
    END_OF_TURN = "end_of_turn"


@dataclass
class EnvAction:
    env_action_type: EnvActionType
    entity_descriptor: EntityDescriptor


class EnvEvents(Enum):
    PLAYER_DEATH = "PLAYER_DEATH"
    OPPONENT_DEATH = "OPPONENT_DEATH"
    WIN_BATTLE = "WIN_BATTLE"


# None, None - observation and actions are not yet implemented
class DeckbuilderSingleBattleEnv(Env[None, None]):
    player: Optional[Player] = None
    opponents: Optional[List[Opponent]] = []

    def __init__(self) -> None:
        self.unemitted_events: List[EnvEvents] = []

    def set_player(self, player: Player) -> None:
        self.player = player

    def set_opponents(self, opponents: List[Opponent]) -> None:
        self.opponents = opponents

    def find_entity_by_descriptor(self, entity_descriptor: EntityDescriptor) -> Entity:
        if entity_descriptor.is_player:
            if self.player is None:
                raise ValueError("Player is not set")
            return self.player
        else:
            if self.opponents is None:
                raise ValueError("Opponents are not set")
            if entity_descriptor.enemy_idx is None:
                raise ValueError("Enemy index is required for non-player entities")
            if (
                entity_descriptor.enemy_idx >= len(self.opponents)
                or entity_descriptor.enemy_idx < 0
            ):
                raise ValueError("Enemy index is out of bounds")
            return self.opponents[entity_descriptor.enemy_idx]

    def get_num_opponents(self) -> int:
        if self.opponents is None:
            raise ValueError("Opponents are not set")
        return len(self.opponents)

    def reduce_entity_hp(
        self, entity_descriptor: EntityDescriptor, amount: int
    ) -> None:
        entity = self.find_entity_by_descriptor(entity_descriptor)
        is_dead = entity.reduce_health(amount)
        if is_dead:
            if entity_descriptor.is_player:
                self.unemitted_events.append(EnvEvents.PLAYER_DEATH)
            else:
                if self.opponents is None:
                    raise ValueError("Opponents are not set")
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
        action: EnvAction | None,
        entity_descriptor: EntityDescriptor,
        trigger_point: EffectTriggerPoint,
    ) -> Optional[EnvAction]:
        entity = self.find_entity_by_descriptor(entity_descriptor)
        entity_active_statuses = entity.get_active_statuses()
        for status_sid in StatusesOrder:
            if status_sid in entity_active_statuses:
                status, amount = entity_active_statuses[status_sid]
                callback = status.get_effects()[trigger_point]
                if callback is not None:
                    action = callback(self, amount, cast(EnvAction, action))
                if action is None:
                    return None  # Action was fully blocked, etc.
        return action

    def attack_entity(self, entity_descriptor: EntityDescriptor, amount: int) -> None:
        action: EnvAction | None = Attack(
            env_action_type=EnvActionType.ATTACK,
            entity_descriptor=entity_descriptor,
            damage=amount,
        )
        action = self.apply_action_callbacks(
            action, entity_descriptor, EffectTriggerPoint.ON_ATTACKED
        )

        if action is not None:
            attack_action = cast(Attack, action)
            self.reduce_entity_hp(entity_descriptor, attack_action.damage)

    def reset_entity_status(
        self, entity_descriptor: EntityDescriptor, status_uid: StatusUIDs
    ) -> None:
        entity = self.find_entity_by_descriptor(entity_descriptor)
        entity.reset_status(status_uid)

    def apply_status_to_entity(
        self, entity_descriptor: EntityDescriptor, status: Status, amount: int
    ) -> None:
        entity = self.find_entity_by_descriptor(entity_descriptor)
        entity.apply_status(status, amount)

    def player_draw_card(self) -> None:
        if self.player is None:
            raise ValueError("Player is not set")
        self.player.draw_card()

    def player_discard_hand(self) -> None:
        if self.player is None:
            raise ValueError("Player is not set")
        self.player.discard_hand()

    def start_turn(self) -> None:
        if self.player is None:
            raise ValueError("Player is not set")
        action = StartOfTurn(
            env_action_type=EnvActionType.START_OF_TURN,
            entity_descriptor=EntityDescriptor(is_player=True),
        )
        self.apply_action_callbacks(
            action,
            EntityDescriptor(is_player=True),
            EffectTriggerPoint.ON_START_OF_TURN,
        )
        if self.opponents is not None:
            for idx in range(len(self.opponents)):
                action = StartOfTurn(
                    env_action_type=EnvActionType.START_OF_TURN,
                    entity_descriptor=EntityDescriptor(is_player=False, enemy_idx=idx),
                )
                self.apply_action_callbacks(
                    action,
                    EntityDescriptor(is_player=False, enemy_idx=idx),
                    EffectTriggerPoint.ON_START_OF_TURN,
                )

    def end_turn(self) -> None:
        if self.player is None:
            raise ValueError("Player is not set")
        action = EndOfTurn(
            env_action_type=EnvActionType.END_OF_TURN,
            entity_descriptor=EntityDescriptor(is_player=True),
        )
        self.apply_action_callbacks(
            action, EntityDescriptor(is_player=True), EffectTriggerPoint.ON_END_OF_TURN
        )
        if self.opponents is not None:
            for idx in range(len(self.opponents)):
                action = EndOfTurn(
                    env_action_type=EnvActionType.END_OF_TURN,
                    entity_descriptor=EntityDescriptor(is_player=False, enemy_idx=idx),
                )
                self.apply_action_callbacks(
                    action,
                    EntityDescriptor(is_player=False, enemy_idx=idx),
                    EffectTriggerPoint.ON_END_OF_TURN,
                )
