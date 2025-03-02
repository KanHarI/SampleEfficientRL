import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from SampleEfficientRL.Envs.Deckbuilder.Card import CardUIDs
from SampleEfficientRL.Envs.Deckbuilder.Opponent import NextMoveType
from SampleEfficientRL.Envs.Deckbuilder.Status import StatusUIDs
from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvTensorizer import (
    ENTITY_TYPE,
    NUM_MAX_ENEMIES,
    SUPPORTED_ENEMY_INTENT_TYPES,
    ActionType,
    SUPPORTED_CARDS_UIDs,
    SUPPORTED_STATUS_UIDs,
    TokenType,
)


class SingleBattleEnvDetensorizer:
    """
    Reverses the tensorization process to reconstruct game states from tensor
    representations, enabling analysis and replay of recorded playthroughs.
    """

    def __init__(self) -> None:
        self.version: str = "1.0"  # Should match the Tensorizer version
        self.header: Optional[Dict[str, Any]] = None
        self.playthrough_data: List[Dict[str, Any]] = []

    def _binary_to_number(
        self, binary_representation: Tuple[int, int, List[int]]
    ) -> int:
        """
        Converts a binary representation back to a number.

        Args:
            binary_representation: A tuple of (scalar_value, sign_bit, binary_list).

        Returns:
            The reconstructed number.
        """
        # For direct access, we could use the scalar_value already stored
        scalar_value, sign_bit, binary_list = binary_representation

        # Alternative: reconstruct from binary
        value = 0
        for i, bit in enumerate(binary_list):
            # Convert from MSB-first (9 to 0) to bit position
            bit_position = 9 - i
            value += bit * (2**bit_position)

        # Apply sign
        if sign_bit == 1:
            value = -value

        return value

    def _decode_card(self, card_encoding: List[int]) -> Optional[CardUIDs]:
        """
        Decodes a one-hot card encoding back to a CardUID.

        Args:
            card_encoding: A one-hot encoded vector representing a card.

        Returns:
            The corresponding CardUID or None if invalid.
        """
        if 1 not in card_encoding:
            return None

        index = card_encoding.index(1)
        if index < len(SUPPORTED_CARDS_UIDs):
            return SUPPORTED_CARDS_UIDs[index]

        return None

    def _decode_status(
        self, status_encoding: List[Any]
    ) -> Tuple[Optional[StatusUIDs], int]:
        """
        Decodes a status encoding back to a StatusUID and value.

        Args:
            status_encoding: A list containing the status one-hot encoding and value.

        Returns:
            A tuple of (StatusUID, value) or (None, 0) if invalid.
        """
        # Split the encoding
        one_hot = status_encoding[:-1]  # All but the last element
        value_binary = status_encoding[-1]

        # Decode status
        if 1 not in one_hot:
            return None, 0

        index = one_hot.index(1)
        status_uid = (
            SUPPORTED_STATUS_UIDs[index] if index < len(SUPPORTED_STATUS_UIDs) else None
        )

        # Decode value
        value = self._binary_to_number(value_binary)

        return status_uid, value

    def _decode_enemy_intent(
        self, intent_encoding: List[Any]
    ) -> Tuple[Optional[NextMoveType], int]:
        """
        Decodes an enemy intent encoding back to a NextMoveType and value.

        Args:
            intent_encoding: A list containing the intent one-hot encoding and value.

        Returns:
            A tuple of (NextMoveType, value) or (None, 0) if invalid.
        """
        # Split the encoding
        one_hot = intent_encoding[:-1]  # All but the last element
        value_binary = intent_encoding[-1]

        # Decode intent type
        if 1 not in one_hot:
            return None, 0

        index = one_hot.index(1)
        intent_type = (
            SUPPORTED_ENEMY_INTENT_TYPES[index]
            if index < len(SUPPORTED_ENEMY_INTENT_TYPES)
            else None
        )

        # Decode value
        value = self._binary_to_number(value_binary)

        return intent_type, value

    def load_playthrough(self, filename: str) -> List[Dict[str, Any]]:
        """
        Loads and deserializes the binary tensor data.

        Args:
            filename: The name of the file to load.

        Returns:
            A list of tensor records.
        """
        self.playthrough_data = []

        with open(filename, "rb") as f:
            # Read header size
            header_size_bytes = f.read(4)
            header_size = int.from_bytes(header_size_bytes, byteorder="little")

            # Read header
            header_bytes = f.read(header_size)
            self.header = json.loads(header_bytes.decode("utf-8"))

            # Process records until end of file
            while True:
                # Read metadata size
                metadata_size_bytes = f.read(4)
                if not metadata_size_bytes:
                    break  # End of file

                metadata_size = int.from_bytes(metadata_size_bytes, byteorder="little")

                # Read metadata
                metadata_bytes = f.read(metadata_size)
                metadata = json.loads(metadata_bytes.decode("utf-8"))

                # Read tensor size
                tensor_size_bytes = f.read(8)
                tensor_size = int.from_bytes(tensor_size_bytes, byteorder="little")

                # Read state data
                state_bytes = f.read(tensor_size)

                # Parse the JSON string to get the state dictionary
                state_dict = json.loads(state_bytes.decode("utf-8"))

                # Create record
                record = {"state": state_dict, **metadata}

                self.playthrough_data.append(record)

        return self.playthrough_data

    def reconstruct_state(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstructs a state dictionary back into a structured, human-readable game state.

        Args:
            state_dict: A dictionary containing state components and turn number.

        Returns:
            A dictionary representing the reconstructed game state.
        """
        # Get state components from the dictionary
        state_components = state_dict.get("state_components", [])
        turn_number = state_dict.get("turn", 0)

        # Initialize state structure
        state = {
            "player": {
                "hp": 0,
                "max_hp": 0,
                "energy": 0,
                "statuses": {},
                "draw_pile": [],
                "hand": [],
                "discard_pile": [],
                "exhaust_pile": [],
            },
            "enemies": [],
            "turn": turn_number,
        }

        # Initialize enemies structure (up to max enemies)
        for i in range(1, NUM_MAX_ENEMIES + 1):
            state["enemies"].append(
                {
                    "idx": i,
                    "hp": 0,
                    "max_hp": 0,
                    "statuses": {},
                    "intent": {"type": None, "value": 0},
                }
            )

        # Process each component
        for component in state_components:
            token_type = component[0]
            entity_or_index = component[1]
            data = component[2]

            # Process by token type
            if token_type == TokenType.ENTITY_HP.value:
                hp_value = self._binary_to_number(data)
                if entity_or_index == ENTITY_TYPE.PLAYER.value:
                    state["player"]["hp"] = hp_value
                elif 1 <= entity_or_index <= NUM_MAX_ENEMIES:
                    state["enemies"][entity_or_index - 1]["hp"] = hp_value

            elif token_type == TokenType.ENTITY_MAX_HP.value:
                max_hp_value = self._binary_to_number(data)
                if entity_or_index == ENTITY_TYPE.PLAYER.value:
                    state["player"]["max_hp"] = max_hp_value
                elif 1 <= entity_or_index <= NUM_MAX_ENEMIES:
                    state["enemies"][entity_or_index - 1]["max_hp"] = max_hp_value

            elif token_type == TokenType.ENTITY_ENERGY.value:
                if entity_or_index == ENTITY_TYPE.PLAYER.value:
                    state["player"]["energy"] = self._binary_to_number(data)

            elif token_type == TokenType.ENTITY_STATUS.value:
                status_uid, value = self._decode_status(data)
                if status_uid:
                    if entity_or_index == ENTITY_TYPE.PLAYER.value:
                        state["player"]["statuses"][status_uid.name] = value
                    elif 1 <= entity_or_index <= NUM_MAX_ENEMIES:
                        state["enemies"][entity_or_index - 1]["statuses"][
                            status_uid.name
                        ] = value

            elif token_type == TokenType.ENEMY_INTENT.value:
                if 1 <= entity_or_index <= NUM_MAX_ENEMIES:
                    intent_type, value = self._decode_enemy_intent(data)
                    if intent_type:
                        state["enemies"][entity_or_index - 1]["intent"] = {
                            "type": intent_type.name,
                            "value": value,
                        }

            elif token_type == TokenType.DRAW_PILE_CARD.value:
                card_uid = self._decode_card(data)
                if card_uid:
                    state["player"]["draw_pile"].append(card_uid.name)

            elif token_type == TokenType.HAND_CARD.value:
                card_uid = self._decode_card(data)
                if card_uid:
                    # Ensure hand has enough slots
                    while len(state["player"]["hand"]) <= entity_or_index:
                        state["player"]["hand"].append(None)
                    state["player"]["hand"][entity_or_index] = card_uid.name

            elif token_type == TokenType.DISCARD_PILE_CARD.value:
                card_uid = self._decode_card(data)
                if card_uid:
                    state["player"]["discard_pile"].append(card_uid.name)

            elif token_type == TokenType.EXHAUST_PILE_CARD.value:
                card_uid = self._decode_card(data)
                if card_uid:
                    state["player"]["exhaust_pile"].append(card_uid.name)

            elif token_type == TokenType.TURN_MARKER.value:
                state["turn"] = self._binary_to_number(data)

        # Clean up enemy list (remove enemies with 0 HP)
        state["enemies"] = [enemy for enemy in state["enemies"] if enemy["hp"] > 0]

        # Remove None values from hand
        state["player"]["hand"] = [
            card for card in state["player"]["hand"] if card is not None
        ]

        return state

    def replay_playthrough(
        self, playthrough_data: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Reconstructs an entire playthrough for analysis.

        Args:
            playthrough_data: A list of records or PlaythroughStep objects. If None, uses the loaded data.

        Returns:
            A list of reconstructed states with their associated actions.
        """
        if playthrough_data is None:
            playthrough_data = self.playthrough_data

        if not playthrough_data:
            return []

        reconstructed_playthrough = []

        for record in playthrough_data:
            # Get the state dictionary - handle both dict and PlaythroughStep
            if hasattr(record, "state"):
                # This is a PlaythroughStep object
                state_dict = record.state
            else:
                # This is a dictionary record
                state_dict = record.get("state", {})

            # Reconstruct state
            reconstructed_state = self.reconstruct_state(state_dict)

            # Get action information
            if hasattr(record, "action_type"):
                # PlaythroughStep object
                action_info = {
                    "action_type": ActionType(record.action_type).name,
                    "reward": record.reward,
                    "turn": record.turn,
                }

                # Add enemy action information if present
                if hasattr(record, "enemy_idx") and record.enemy_idx is not None:
                    action_info["enemy_idx"] = record.enemy_idx
                    action_info["move_type"] = record.move_type
                    action_info["amount"] = record.amount
            else:
                # Dictionary record
                action_info = {
                    "action_type": ActionType(record["action_type"]).name,
                    "reward": record["reward"],
                    "turn": record["turn"],
                }

                # Add enemy action information if present
                if record.get("enemy_idx") is not None:
                    action_info["enemy_idx"] = record["enemy_idx"]
                    action_info["move_type"] = record["move_type"]
                    action_info["amount"] = record["amount"]

            # Create a combined record with the state and action info
            combined_record = {"state": reconstructed_state, **action_info}
            reconstructed_playthrough.append(combined_record)

        return reconstructed_playthrough

    def decode_playthrough(
        self, playthrough_data: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Alias for replay_playthrough for backward compatibility.

        Args:
            playthrough_data: A list of tensor records. If None, uses the loaded data.

        Returns:
            A list of reconstructed states with their associated actions.
        """
        return self.replay_playthrough(playthrough_data)
