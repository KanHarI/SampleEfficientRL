import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch

from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvTensorizer import (
    BINARY_NUMBER_BITS,
    CURRENT_TENSORIZER_VERSION,
    MAX_ENCODED_NUMBER,
    NUMBER_ENCODING_DIMS,
    SUPPORTED_ENEMY_INTENT_TYPES,
    ActionType,
    PlaythroughStep,
    ReplayMetadata,
    SUPPORTED_CARDS_UIDs,
    SUPPORTED_STATUS_UIDs,
    TokenType,
)


def print_separator() -> None:
    """Print a separator line."""
    print("\n" + "=" * 40 + "\n")


class ReplayExplorer:
    def __init__(self, replay_path: str):
        """
        Initialize the replay explorer with a path to a replay file.

        Args:
            replay_path: Path to the replay file
        """
        self.replay_path = replay_path
        # Add PlaythroughStep and ReplayMetadata to safe globals for loading
        torch.serialization.add_safe_globals(
            [PlaythroughStep, ActionType, ReplayMetadata]
        )

        # Load the playthrough data
        raw_data = torch.load(replay_path, weights_only=False)

        # Check data format and extract playthrough data and metadata
        self.metadata: Optional[ReplayMetadata] = None
        if (
            isinstance(raw_data, dict)
            and "metadata" in raw_data
            and "playthrough_steps" in raw_data
        ):
            # New format with metadata
            self.metadata = raw_data["metadata"]
            loaded_data = raw_data["playthrough_steps"]

            # Check version compatibility if required
            if self.metadata.version != CURRENT_TENSORIZER_VERSION:
                print(
                    f"WARNING: Replay version ({self.metadata.version}) doesn't match required version ({required_version})"
                )
                print("This replay may not be compatible with the current code.")
        else:
            # Legacy format without metadata
            loaded_data = raw_data
            print(
                "WARNING: This replay file uses the legacy format without version metadata."
            )

        # Determine data format of playthrough steps (PlaythroughStep objects or just state tuples)
        self.is_simple_tuple_format = isinstance(loaded_data[0], tuple)

        if self.is_simple_tuple_format:
            # Simple tuple format: create artificial PlaythroughStep objects with NO_OP actions
            self.playthrough_data = [
                PlaythroughStep(
                    state=state_tuple,
                    action_type=ActionType.NO_OP,
                    card_idx=None,
                    target_idx=None,
                    reward=0.0,
                )
                for state_tuple in loaded_data
            ]
        else:
            # Standard format with PlaythroughStep objects
            self.playthrough_data = cast(List[PlaythroughStep], loaded_data)

        self.total_steps = len(self.playthrough_data)

        # Mapping dictionaries for readable output
        self.card_uid_map: Dict[int, str] = {
            i + 1: uid.name for i, uid in enumerate(SUPPORTED_CARDS_UIDs)
        }
        self.status_uid_map: Dict[int, str] = {
            i + 1: uid.name for i, uid in enumerate(SUPPORTED_STATUS_UIDs)
        }
        self.intent_type_map: Dict[int, str] = {
            i + 1: intent.name for i, intent in enumerate(SUPPORTED_ENEMY_INTENT_TYPES)
        }

        print(f"Loaded replay with {self.total_steps} steps from {replay_path}")
        if self.metadata:
            timestamp_str = (
                time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(self.metadata.timestamp)
                )
                if self.metadata.timestamp
                else "Unknown"
            )
            print(
                f"Replay version: {self.metadata.version}, Recorded on: {timestamp_str}"
            )
            if self.metadata.notes:
                print(f"Notes: {self.metadata.notes}")

        if self.is_simple_tuple_format:
            print("NOTE: This is a simple state-only replay without action information")

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

    def _decode_state(self, step: PlaythroughStep) -> Dict[str, Any]:
        """
        Decode a step's state tensors into a readable format.

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

        # Initialize state container
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

        # Process each token in the state
        current_enemy: Optional[Dict[str, Any]] = None

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
                    # First few cards are considered hand, if there are no specific tokens for hand
                    if (
                        len(state["player"]["hand"]) < 5
                        and len(state["player"]["draw_pile"]) == 0
                    ):
                        state["player"]["hand"].append(
                            {
                                "name": card_name,
                                "cost": 1,  # Default cost, we don't have actual cost in tensor
                            }
                        )
                    else:
                        state["player"]["draw_pile"].append(
                            {
                                "name": card_name,
                                "cost": 1,  # Default cost, we don't have actual cost in tensor
                            }
                        )

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

            # Handle entity HP
            elif token_type == TokenType.ENTITY_HP.value:
                hp_value = self._extract_numeric_value(encoded_numbers[i])
                # If we don't have player HP yet, this is player HP
                if state["player"]["hp"] == 0:
                    state["player"]["hp"] = hp_value
                else:
                    # Create a new enemy if needed
                    if current_enemy is None:
                        current_enemy = {
                            "hp": 0,
                            "max_hp": 0,
                            "intent": None,
                            "statuses": {},
                        }
                    current_enemy["hp"] = hp_value

            # Handle entity max HP
            elif token_type == TokenType.ENTITY_MAX_HP.value:
                max_hp_value = self._extract_numeric_value(encoded_numbers[i])
                # If we don't have player max HP yet, this is player max HP
                if state["player"]["max_hp"] == 0:
                    state["player"]["max_hp"] = max_hp_value
                elif current_enemy is not None:
                    current_enemy["max_hp"] = max_hp_value
                    # Add enemy to list if we have all required info
                    if current_enemy["hp"] > 0 and current_enemy["max_hp"] > 0:
                        state["enemies"].append(current_enemy)
                        current_enemy = None

            # Handle entity energy
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
                    # If we don't have enemy yet or have added it to list, this is player status
                    if current_enemy is None or current_enemy in state["enemies"]:
                        player_statuses = state["player"]["statuses"]
                        player_statuses[status_name] = status_amount
                    else:
                        if current_enemy is not None:
                            enemy_statuses = current_enemy["statuses"]
                            enemy_statuses[status_name] = status_amount

            # Handle enemy intent
            elif token_type == TokenType.ENEMY_INTENT.value:
                intent_idx = int(enemy_intent_indices[i].item())
                intent_amount = self._extract_numeric_value(encoded_numbers[i])
                if intent_idx > 0 and current_enemy is not None:
                    intent_name = self.intent_type_map.get(
                        intent_idx, f"Unknown({intent_idx})"
                    )
                    current_enemy["intent"] = {
                        "name": intent_name,
                        "amount": intent_amount,
                    }

        # Make sure all enemies are added
        if current_enemy is not None and current_enemy not in state["enemies"]:
            state["enemies"].append(current_enemy)

        return state

    def print_state_summary(self, step_idx: int, decoded_state: Dict[str, Any]) -> None:
        """Print a summary of the state for one step."""
        # Player info
        player = decoded_state["player"]
        print(
            f"Player HP: {player['hp']}/{player['max_hp']}, Energy: {player['energy']}"
        )

        # Enemies info
        for i, enemy in enumerate(decoded_state["enemies"]):
            print(f"Opponent {i+1} HP: {enemy['hp']}/{enemy['max_hp']}")
            if enemy["intent"]:
                print(
                    f"  Intent: {enemy['intent']['name']} with amount {enemy['intent']['amount']}"
                )

    def print_action_summary(
        self, step_idx: int, decoded_state: Dict[str, Any]
    ) -> None:
        """Print a summary of the action for one step."""
        action = decoded_state["action"]
        action_type = action["type"]

        if action_type == "PLAY_CARD" and action["card_idx"] is not None:
            card_idx = action["card_idx"]
            target_idx = action["target_idx"]
            player_hand = decoded_state["player"]["hand"]
            if card_idx < len(player_hand):
                card_name = player_hand[card_idx]["name"]
                print(
                    f"  Action: Play card {card_name} (index {card_idx}), targeting enemy {target_idx}"
                )
            else:
                print(
                    f"  Action: Play card at index {card_idx}, targeting enemy {target_idx}"
                )
        elif action_type == "END_TURN":
            print("  Action: End Turn")
        elif action_type == "NO_OP":
            print("  Action: No Operation")

    def print_full_replay(self) -> None:
        """Print the entire replay in a turn-by-turn format."""
        print_separator()
        print("FULL REPLAY")
        print_separator()

        # For simple state format without action information, use a simplified display
        if self.is_simple_tuple_format:
            for step_idx in range(self.total_steps):
                print(f"STATE {step_idx + 1}")
                print_separator()

                step = self.playthrough_data[step_idx]
                decoded_state = self._decode_state(step)

                self.print_state_summary(step_idx, decoded_state)
                print_separator()

            print("END OF REPLAY")
            print_separator()
            return

        # For full PlaythroughStep format with action information, use the detailed display
        current_turn = 1
        player_turn = True
        in_enemy_phase = False

        # First step is always start of turn 1
        print(f"START TURN {current_turn}")
        print_separator()

        for step_idx in range(self.total_steps):
            step = self.playthrough_data[step_idx]
            decoded_state = self._decode_state(step)
            action_type = step.action_type

            # Print state at the start of each turn or phase
            if step_idx == 0 or (
                step_idx > 0
                and (
                    # Start of player turn (we just detected a new turn)
                    (player_turn and not in_enemy_phase)
                    or
                    # Start of enemy phase
                    (
                        in_enemy_phase
                        and step_idx > 0
                        and self.playthrough_data[step_idx - 1].action_type
                        == ActionType.END_TURN
                        and action_type == ActionType.NO_OP
                    )
                )
            ):
                self.print_state_summary(step_idx, decoded_state)
                print("")

            # Print the action
            self.print_action_summary(step_idx, decoded_state)

            # Detect phase transitions
            if action_type == ActionType.END_TURN:
                if player_turn and not in_enemy_phase:
                    # Player turn ending, moving to enemy phase
                    print(f"END OF PLAYER PHASE (TURN {current_turn})")
                    print("")
                    in_enemy_phase = True

                    # Print enemy phase header if there's a next step and it's a NO_OP
                    if (
                        step_idx + 1 < self.total_steps
                        and self.playthrough_data[step_idx + 1].action_type
                        == ActionType.NO_OP
                    ):
                        print(f"ENEMY PHASE (TURN {current_turn})")

                elif in_enemy_phase:
                    # Enemy phase ending, moving to next player turn
                    print("")
                    in_enemy_phase = False
                    player_turn = True
                    current_turn += 1

                    # Print next turn header if there's a next step
                    if step_idx + 1 < self.total_steps:
                        print(f"START TURN {current_turn}")
                        print_separator()

        print_separator()
        print("END OF REPLAY")
        print_separator()


def main() -> None:
    """Main entry point for the replay explorer."""
    parser = argparse.ArgumentParser(
        description="Explore a saved tensor replay as text."
    )
    parser.add_argument("replay_path", type=str, help="Path to the replay file")
    parser.add_argument("--version", type=str, help="Required replay version")
    args = parser.parse_args()

    # Check if the replay file exists
    if not os.path.exists(args.replay_path):
        print(f"Error: Replay file '{args.replay_path}' not found.")
        return

    try:
        explorer = ReplayExplorer(args.replay_path, args.version)
        explorer.print_full_replay()
    except Exception as e:
        print(f"Error loading or exploring replay: {e}")


if __name__ == "__main__":
    main()
