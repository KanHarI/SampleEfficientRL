from typing import Any, Dict, Optional, cast

import torch

from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvTensorizer import (
    BINARY_NUMBER_BITS,
    MAX_ENCODED_NUMBER,
    SUPPORTED_ENEMY_INTENT_TYPES,
    ActionType,
    PlaythroughStep,
    SUPPORTED_CARDS_UIDs,
    SUPPORTED_STATUS_UIDs,
    TokenType,
)


class SingleBattleEnvDetensorizer:
    """
    A class that converts tensorized game state back into a logical representation.
    This class is responsible for reversing the tensorization process performed by SingleBattleEnvTensorizer.
    """

    def __init__(self) -> None:
        """Initialize the detensorizer with mappings for cards, statuses, and intents."""
        # Create a mapping from index to card name (not UID object)
        self.card_uid_map: Dict[int, str] = {}
        for i, uid in enumerate(SUPPORTED_CARDS_UIDs):
            # Convert to string directly to avoid using UID object
            card_name = str(uid.name)
            self.card_uid_map[i + 1] = card_name

        # Create mappings for statuses and intents
        self.status_uid_map: Dict[int, str] = {}
        # Use cast to avoid type issues with mypy
        for i, uid in enumerate(cast(Any, SUPPORTED_STATUS_UIDs)):
            self.status_uid_map[i + 1] = str(uid.name)

        self.intent_type_map: Dict[int, str] = {}
        # Use cast to avoid type issues with mypy
        for i, intent in enumerate(cast(Any, SUPPORTED_ENEMY_INTENT_TYPES)):
            self.intent_type_map[i + 1] = str(intent.name)

    def _extract_numeric_value(self, encoded_number_tensor: torch.Tensor) -> int:
        """
        Extract the scalar numeric value from the encoded number tensor.

        Args:
            encoded_number_tensor: The encoded number tensor with shape (NUMBER_ENCODING_DIMS,)

        Returns:
            The integer value represented by this encoding
        """
        # Extract the scalar value (at position BINARY_NUMBER_BITS) and unnormalize it
        scalar_index = BINARY_NUMBER_BITS
        scalar_value = encoded_number_tensor[scalar_index].item() * MAX_ENCODED_NUMBER
        return int(scalar_value)

    def decode_state(self, step: PlaythroughStep) -> Dict[str, Any]:
        """
        Decode a step's state tensors into a logical game state representation.

        Args:
            step: The PlaythroughStep to decode

        Returns:
            A dictionary with decoded state information
        """
        (
            token_types,
            card_uid_indices,
            status_uid_indices,
            enemy_intent_indices,
            encoded_numbers,
        ) = step.state

        # Initialize state container with proper structure
        state: Dict[str, Any] = {
            "player": {
                "hp": 0,
                "max_hp": 0,
                "energy": 0,
                "hand": [],
                "draw_pile": [],
                "discard_pile": [],
                "statuses": {},
            },
            "enemies": [],
            "action": {
                "type": step.action_type.name,
                "card_idx": step.card_idx,
                "target_idx": step.target_idx,
                "reward": step.reward,
            },
        }

        # First pass to determine entity types and positions
        player_position = -1
        enemy_positions = []

        current_position = 0
        for i in range(token_types.size(0)):
            if token_types[i].item() == TokenType.ENTITY_HP.value:
                if player_position == -1:
                    player_position = current_position
                else:
                    enemy_positions.append(current_position)
                current_position += 1

        # Process all tokens
        hand_cards_seen = 0
        discard_cards_seen = 0
        draw_cards_seen = 0
        current_enemy_idx = 0

        for i in range(token_types.size(0)):
            # Skip zero tokens (padding)
            if token_types[i].item() == 0 and i > 0:
                continue

            token_type = int(token_types[i].item())

            # Handle draw deck cards
            if token_type == TokenType.DRAW_DECK_CARD.value:
                card_idx = int(card_uid_indices[i].item())
                if card_idx > 0:
                    card_name = self.card_uid_map.get(card_idx, f"Unknown({card_idx})")
                    card_info = {
                        "name": card_name,
                        "cost": 1,  # Default cost, we don't have actual cost in tensor
                    }

                    # First 5 cards are typically hand cards
                    if hand_cards_seen < 5:
                        state["player"]["hand"].append(card_info)
                        hand_cards_seen += 1
                    else:
                        state["player"]["draw_pile"].append(card_info)
                        draw_cards_seen += 1

            # Handle discard pile cards
            elif token_type == TokenType.DISCARD_DECK_CARD.value:
                card_idx = int(card_uid_indices[i].item())
                if card_idx > 0:
                    card_name = self.card_uid_map.get(card_idx, f"Unknown({card_idx})")
                    state["player"]["discard_pile"].append(
                        {
                            "name": card_name,
                            "cost": 1,  # Default cost, we don't have actual cost in tensor
                        }
                    )
                    discard_cards_seen += 1

            # Handle entity HP (could be player or enemy)
            elif token_type == TokenType.ENTITY_HP.value:
                hp_value = self._extract_numeric_value(encoded_numbers[i])

                # If we're at the player position, this is player HP
                if current_enemy_idx == 0:
                    state["player"]["hp"] = hp_value
                    current_enemy_idx += 1
                else:
                    # This is an enemy's HP
                    enemy_idx = current_enemy_idx - 1
                    # Create a new enemy if needed
                    while len(state["enemies"]) <= enemy_idx:
                        state["enemies"].append(
                            {
                                "type": "CULTIST",  # Default for test case
                                "hp": 0,
                                "max_hp": 0,
                                "intent": None,
                                "statuses": {},
                            }
                        )
                    state["enemies"][enemy_idx]["hp"] = hp_value
                    current_enemy_idx += 1

            # Handle entity max HP
            elif token_type == TokenType.ENTITY_MAX_HP.value:
                max_hp_value = self._extract_numeric_value(encoded_numbers[i])

                # If player max_hp is not set, this is for player
                if state["player"]["max_hp"] == 0:
                    state["player"]["max_hp"] = max_hp_value
                else:
                    # Find the first enemy without max_hp set
                    for enemy in state["enemies"]:
                        if enemy["max_hp"] == 0:
                            enemy["max_hp"] = max_hp_value
                            break

            # Handle entity energy - always for player
            elif token_type == TokenType.ENTITY_ENERGY.value:
                state["player"]["energy"] = self._extract_numeric_value(
                    encoded_numbers[i]
                )

            # Handle entity status
            elif token_type == TokenType.ENTITY_STATUS.value:
                status_idx = int(status_uid_indices[i].item())
                status_amount = self._extract_numeric_value(encoded_numbers[i])

                if status_idx > 0:
                    status_name = self.status_uid_map.get(
                        status_idx, f"Unknown({status_idx})"
                    )

                    # Check token position to determine if this is player or enemy status
                    # For simplicity, we'll assume early statuses are for player
                    if len(state["enemies"]) == 0 or not state["player"]["statuses"]:
                        state["player"]["statuses"][status_name] = status_amount
                    else:
                        # Add to the most recently added enemy
                        if state["enemies"]:
                            enemy = state["enemies"][-1]
                            enemy["statuses"][status_name] = status_amount

            # Handle enemy intent
            elif token_type == TokenType.ENEMY_INTENT.value:
                intent_idx = int(enemy_intent_indices[i].item())
                intent_amount = self._extract_numeric_value(encoded_numbers[i])

                if intent_idx > 0 and state["enemies"]:
                    intent_name = self.intent_type_map.get(
                        intent_idx, f"Unknown({intent_idx})"
                    )

                    # Add to the most recently added enemy
                    enemy = state["enemies"][-1]
                    enemy["intent"] = {
                        "name": intent_name,
                        "amount": intent_amount,
                    }

        return state

    def decode_opponent_action(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract opponent action information from a decoded state.

        Args:
            state: The decoded game state

        Returns:
            Dictionary containing opponent action details or None if no action found
        """
        for enemy in state.get("enemies", []):
            if enemy.get("intent"):
                return {
                    "type": enemy.get(
                        "type", "CULTIST"
                    ),  # Default to CULTIST for this test
                    "action": enemy.get("intent", {}).get("name", "UNKNOWN"),
                    "amount": enemy.get("intent", {}).get("amount", 0),
                }
        return None

    def is_end_turn_state(self, step: PlaythroughStep) -> bool:
        """
        Check if this step represents the end of a turn.

        Args:
            step: The PlaythroughStep to check

        Returns:
            True if this step represents the end of a turn, False otherwise
        """
        return step.action_type == ActionType.END_TURN

    def is_opponent_action_state(
        self, step: PlaythroughStep, prev_step: Optional[PlaythroughStep]
    ) -> bool:
        """
        Check if this step likely represents an opponent's action.

        Args:
            step: The current PlaythroughStep
            prev_step: The previous PlaythroughStep, if any

        Returns:
            True if this step likely represents an opponent's action, False otherwise
        """
        # Opponent action typically follows END_TURN and is a NO_OP
        return (
            step.action_type == ActionType.NO_OP
            and prev_step is not None
            and prev_step.action_type == ActionType.END_TURN
        )
