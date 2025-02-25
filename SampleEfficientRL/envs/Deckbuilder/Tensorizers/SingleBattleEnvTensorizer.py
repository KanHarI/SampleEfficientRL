import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch

from SampleEfficientRL.Envs.Deckbuilder.Card import CardUIDs
from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
    DeckbuilderSingleBattleEnv,
)
from SampleEfficientRL.Envs.Deckbuilder.Opponent import NextMoveType
from SampleEfficientRL.Envs.Deckbuilder.Status import StatusUIDs

# Warning: changing this list will invalidate all the pre-trained models weights
SUPPORTED_CARDS_UIDs: List[CardUIDs] = [
    CardUIDs.BASH,
    CardUIDs.STRIKE,
    CardUIDs.DEFEND,
]

SUPPORTED_STATUS_UIDs: List[StatusUIDs] = [
    StatusUIDs.BLOCK,
    StatusUIDs.VULNERABLE,
    StatusUIDs.RITUAL,
    StatusUIDs.STRENGTH,
    StatusUIDs.ENERGY_USER,
    StatusUIDs.HAND_DRAWER,
]

SUPPORTED_ENEMY_INTENT_TYPES: List[NextMoveType] = [
    NextMoveType.ATTACK,
    NextMoveType.RITUAL,
]


class ENTITY_TYPE(Enum):
    PLAYER = 0
    ENEMY_1 = 1
    ENEMY_2 = 2
    ENEMY_3 = 3
    ENEMY_4 = 4
    ENEMY_5 = 5
    ENEMY_6 = 6


class TokenType(Enum):
    DRAW_DECK_CARD = 0
    DISCARD_DECK_CARD = 1
    EXHAUST_DECK_CARD = 2
    ENTITY_HP = 3
    ENTITY_MAX_HP = 4
    ENTITY_ENERGY = 5
    ENTITY_STATUS = 5
    ENEMY_INTENT = 6


class ActionType(Enum):
    PLAY_CARD = 0
    END_TURN = 1
    NO_OP = 2  # For states where no action is taken


NUM_MAX_ENEMIES = ENTITY_TYPE.ENEMY_6.value

MAX_ENCODED_NUMBER = 1023
BINARY_NUMBER_BITS = 10
SCALAR_NUMBER_DIMS = 1
LOG_NUMBER_DIMS = 1
NUMBER_ENCODING_DIMS = BINARY_NUMBER_BITS + SCALAR_NUMBER_DIMS + LOG_NUMBER_DIMS


class TensorizerMode(Enum):
    OBSERVE = 0  # Just observe current game state
    RECORD = 1  # Record playthrough with actions


@dataclass
class SingleBattleEnvTensorizerConfig:
    """
    Input for this tensorizer is the current state of the environment
    Output is (savable, serializable) tensor representation of the state
    """

    context_size: int
    mode: TensorizerMode = TensorizerMode.OBSERVE


@dataclass
class PlaythroughStep:
    """A single step in a playthrough, containing state and action information."""

    state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    action_type: ActionType
    card_idx: Optional[int] = None
    target_idx: Optional[int] = None
    reward: float = 0.0


class SingleBattleEnvTensorizer:
    def __init__(self, config: SingleBattleEnvTensorizerConfig):
        self.config = config
        self.playthrough_steps: List[PlaythroughStep] = []

    def _encode_number(self, num: int) -> torch.Tensor:
        """
        Encodes a number into a 12-dimensional representation:
        - 10 binary bits
        - The scalar value (normalized)
        - The log (base e) of the scalar value (or -1 for 0)

        Args:
            num: The number to encode

        Returns:
            A tensor with shape (NUMBER_ENCODING_DIMS,) containing the encoded number
        """
        # Cap the number
        num = min(num, MAX_ENCODED_NUMBER)

        # Initialize tensor for the encoded number
        encoded = torch.zeros(NUMBER_ENCODING_DIMS, dtype=torch.float)

        # Binary bits (10 bits)
        for i in range(BINARY_NUMBER_BITS):
            if num & (1 << i):
                encoded[i] = 1.0

        # Scalar value (normalized to [0, 1])
        encoded[BINARY_NUMBER_BITS] = float(num) / MAX_ENCODED_NUMBER

        # Log value
        if num > 0:
            encoded[BINARY_NUMBER_BITS + SCALAR_NUMBER_DIMS] = math.log(float(num))
        else:
            encoded[BINARY_NUMBER_BITS + SCALAR_NUMBER_DIMS] = -1.0

        return encoded

    # Return tensors tuple:
    # Token types,
    # Card uid indices,
    # Status uid indices,
    # Enemy intent type indices,
    # Encoded numbers,
    def tensorize(
        self, state: DeckbuilderSingleBattleEnv
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts the current state of the environment into a tensor representation.

        Args:
            state: The current state of the environment.

        Returns:
            A tuple of tensors:
            - token_types: The types of tokens in the context.
            - card_uid_indices: The indices of card UIDs in the context.
            - status_uid_indices: The indices of status UIDs in the context.
            - enemy_intent_indices: The indices of enemy intent types in the context.
            - encoded_numbers: The encoded numerical values in the context.

        Raises:
            ValueError: If the state representation exceeds the configured context size.
        """
        # Initialize tensors with zeros
        token_types = torch.zeros(self.config.context_size, dtype=torch.long)
        card_uid_indices = torch.zeros(self.config.context_size, dtype=torch.long)
        status_uid_indices = torch.zeros(self.config.context_size, dtype=torch.long)
        enemy_intent_indices = torch.zeros(self.config.context_size, dtype=torch.long)
        encoded_numbers = torch.zeros(
            (self.config.context_size, NUMBER_ENCODING_DIMS), dtype=torch.float
        )

        position = 0

        def check_context_size() -> None:
            nonlocal position
            if position >= self.config.context_size:
                raise ValueError(
                    f"State representation exceeds configured context size of {self.config.context_size}"
                )

        # Encode player's draw pile
        player = state.player
        if player is None:
            raise ValueError("Player is not set")

        for card in player.draw_pile:
            check_context_size()

            token_types[position] = TokenType.DRAW_DECK_CARD.value
            if card.card_uid in SUPPORTED_CARDS_UIDs:
                card_uid_indices[position] = (
                    SUPPORTED_CARDS_UIDs.index(card.card_uid) + 1
                )  # +1 for padding

            position += 1

        # Encode player's discard pile
        for card in player.discard_pile:
            check_context_size()

            token_types[position] = TokenType.DISCARD_DECK_CARD.value
            if card.card_uid in SUPPORTED_CARDS_UIDs:
                card_uid_indices[position] = (
                    SUPPORTED_CARDS_UIDs.index(card.card_uid) + 1
                )

            position += 1

        # Encode player's hand
        for card in player.hand:
            check_context_size()

            # Using draw deck token type for hand cards since there's no specific hand token
            token_types[position] = TokenType.DRAW_DECK_CARD.value
            if card.card_uid in SUPPORTED_CARDS_UIDs:
                card_uid_indices[position] = (
                    SUPPORTED_CARDS_UIDs.index(card.card_uid) + 1
                )

            position += 1

        # Encode player entity information
        check_context_size()
        token_types[position] = TokenType.ENTITY_HP.value
        encoded_numbers[position] = self._encode_number(player.current_health)
        position += 1

        check_context_size()
        token_types[position] = TokenType.ENTITY_MAX_HP.value
        encoded_numbers[position] = self._encode_number(player.max_health)
        position += 1

        check_context_size()
        token_types[position] = TokenType.ENTITY_ENERGY.value
        encoded_numbers[position] = self._encode_number(player.energy)
        position += 1

        # Encode player statuses
        for status_uid, (status, amount) in player.get_active_statuses().items():
            check_context_size()

            if status_uid in SUPPORTED_STATUS_UIDs:
                token_types[position] = TokenType.ENTITY_STATUS.value
                status_uid_indices[position] = (
                    SUPPORTED_STATUS_UIDs.index(status_uid) + 1
                )
                encoded_numbers[position] = self._encode_number(amount)
                position += 1

        # Encode enemies
        if state.opponents is None:
            raise ValueError("Opponents not set")

        for enemy_idx, enemy in enumerate(state.opponents):
            if enemy.current_health > 0:  # Check if enemy is alive using current_health
                # Enemy HP
                check_context_size()
                token_types[position] = TokenType.ENTITY_HP.value
                encoded_numbers[position] = self._encode_number(enemy.current_health)
                position += 1

                # Enemy Max HP
                check_context_size()
                token_types[position] = TokenType.ENTITY_MAX_HP.value
                encoded_numbers[position] = self._encode_number(enemy.max_health)
                position += 1

                # Enemy intent
                if enemy.next_move:
                    check_context_size()
                    token_types[position] = TokenType.ENEMY_INTENT.value
                    if enemy.next_move.move_type in SUPPORTED_ENEMY_INTENT_TYPES:
                        enemy_intent_indices[position] = (
                            SUPPORTED_ENEMY_INTENT_TYPES.index(
                                enemy.next_move.move_type
                            )
                            + 1
                        )
                    if enemy.next_move.amount is not None:
                        encoded_numbers[position] = self._encode_number(
                            enemy.next_move.amount
                        )
                    position += 1

                # Enemy statuses
                for status_uid, (status, amount) in enemy.get_active_statuses().items():
                    check_context_size()

                    if status_uid in SUPPORTED_STATUS_UIDs:
                        token_types[position] = TokenType.ENTITY_STATUS.value
                        status_uid_indices[position] = (
                            SUPPORTED_STATUS_UIDs.index(status_uid) + 1
                        )
                        encoded_numbers[position] = self._encode_number(amount)
                        position += 1

        return (
            token_types,
            card_uid_indices,
            status_uid_indices,
            enemy_intent_indices,
            encoded_numbers,
        )

    def record_action(
        self,
        state_tensor: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        action_type: ActionType,
        card_idx: Optional[int] = None,
        target_idx: Optional[int] = None,
        reward: float = 0.0,
    ) -> None:
        """
        Record an action taken in the given state.

        Args:
            state_tensor: The tensor representation of the state before the action.
            action_type: The type of action taken.
            card_idx: The index of the card played (if applicable).
            target_idx: The index of the target for the card (if applicable).
            reward: The reward received for taking this action.
        """
        if self.config.mode != TensorizerMode.RECORD:
            return

        step = PlaythroughStep(
            state=state_tensor,
            action_type=action_type,
            card_idx=card_idx,
            target_idx=target_idx,
            reward=reward,
        )
        self.playthrough_steps.append(step)

    def get_playthrough_data(self) -> List[PlaythroughStep]:
        """Get the recorded playthrough data."""
        return self.playthrough_steps

    def clear_playthrough_data(self) -> None:
        """Clear the recorded playthrough data."""
        self.playthrough_steps = []

    def record_play_card(
        self,
        state: DeckbuilderSingleBattleEnv,
        card_idx: int,
        target_idx: int,
        reward: float = 0.0,
    ) -> None:
        """
        Record a 'play card' action.

        Args:
            state: The current game state.
            card_idx: The index of the card played.
            target_idx: The index of the target.
            reward: The reward received for this action.
        """
        if self.config.mode != TensorizerMode.RECORD:
            return

        state_tensor = self.tensorize(state)
        self.record_action(
            state_tensor=state_tensor,
            action_type=ActionType.PLAY_CARD,
            card_idx=card_idx,
            target_idx=target_idx,
            reward=reward,
        )

    def record_end_turn(
        self, state: DeckbuilderSingleBattleEnv, reward: float = 0.0
    ) -> None:
        """
        Record an 'end turn' action.

        Args:
            state: The current game state.
            reward: The reward received for this action.
        """
        if self.config.mode != TensorizerMode.RECORD:
            return

        state_tensor = self.tensorize(state)
        self.record_action(
            state_tensor=state_tensor, action_type=ActionType.END_TURN, reward=reward
        )

    def save_playthrough(self, filename: str) -> None:
        """
        Save the recorded playthrough data to a file.

        Args:
            filename: The path to save the data to.
        """
        torch.save(self.playthrough_steps, filename)
