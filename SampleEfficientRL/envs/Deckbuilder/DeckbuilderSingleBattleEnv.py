from enum import Enum
from typing import List, Optional, cast

import SampleEfficientRL.Envs.Deckbuilder.EnvActions.Attack as AttackModule
from SampleEfficientRL.Envs.Deckbuilder.Card import Card, CardEffectTrigger
from SampleEfficientRL.Envs.Deckbuilder.Entity import Entity
from SampleEfficientRL.Envs.Deckbuilder.EnvAction import (
    EntityDescriptor,
    EnvAction,
    EnvActionType,
)
from SampleEfficientRL.Envs.Deckbuilder.EnvActions.EndOfTurn import EndOfTurn
from SampleEfficientRL.Envs.Deckbuilder.EnvActions.StartOfTurn import StartOfTurn
from SampleEfficientRL.Envs.Deckbuilder.Opponent import Opponent
from SampleEfficientRL.Envs.Deckbuilder.Player import PlayCardResult, Player
from SampleEfficientRL.Envs.Deckbuilder.Status import (
    EffectTriggerPoint,
    Status,
    StatusesOrder,
    StatusUIDs,
)
from SampleEfficientRL.Envs.Env import Env


class EnvEvents(Enum):
    PLAYER_DEATH = "PLAYER_DEATH"
    OPPONENT_DEATH = "OPPONENT_DEATH"
    WIN_BATTLE = "WIN_BATTLE"


# None, None - observation and actions are not yet implemented
class DeckbuilderSingleBattleEnv(Env[None, None]):
    player: Optional[Player] = None
    opponents: Optional[List[Opponent]] = []
    num_turn: int = 0

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
                if trigger_point in status.get_effects():
                    callback = status.get_effects()[trigger_point]
                    if callback is not None:
                        action = callback(self, amount, cast(EnvAction, action))
                if action is None:
                    return None  # Action was fully blocked, etc.
        return action

    def attack_entity(
        self, source: EntityDescriptor, target: EntityDescriptor, amount: int
    ) -> None:
        action: EnvAction | None = AttackModule.Attack(
            env_action_type=EnvActionType.ATTACK,
            entity_descriptor=target,
            damage=amount,
        )
        action = self.apply_action_callbacks(
            action, source, EffectTriggerPoint.ON_ATTACK
        )
        action = self.apply_action_callbacks(
            action, target, EffectTriggerPoint.ON_ATTACKED
        )

        if action is not None:
            attack_action = cast(AttackModule.Attack, action)
            self.reduce_entity_hp(target, attack_action.damage)

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
            for idx in range(len(self.opponents)):
                self.opponents[idx].start_turn()

    def end_turn(self) -> None:
        if self.player is None:
            raise ValueError("Player is not set")
        if self.opponents is None:
            raise ValueError("Opponents are not set")
        for idx in range(len(self.opponents)):
            self.opponents[idx].perform_next_move(
                EntityDescriptor(is_player=False, enemy_idx=idx),
            )
        action = EndOfTurn(
            env_action_type=EnvActionType.END_OF_TURN,
            entity_descriptor=EntityDescriptor(is_player=True),
        )
        self.apply_action_callbacks(
            action, EntityDescriptor(is_player=True), EffectTriggerPoint.ON_END_OF_TURN
        )
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
        self.num_turn += 1

    def play_card(self, card: Card, target_idx: Optional[int] = None) -> None:
        card_callbacks = card.get_effects()
        if CardEffectTrigger.ON_PLAY in card_callbacks:
            callback = card_callbacks[CardEffectTrigger.ON_PLAY]
            callback(self, target_idx)

    def play_card_from_hand(
        self, card_idx: int, target_idx: Optional[int] = None
    ) -> PlayCardResult:
        if self.player is None:
            raise ValueError("Player not defined")
        return self.player.play_card(card_idx, target_idx)

    def reset(self) -> None:
        pass

    def step(self, action: None) -> None:
        pass

    def observe(self) -> None:
        pass
