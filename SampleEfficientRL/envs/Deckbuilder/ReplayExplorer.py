import argparse
import os
from typing import Any, Dict, List, cast

import torch

from SampleEfficientRL.Envs.Deckbuilder.GameOutputManager import GameOutputManager
from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvDetensorizer import (
    SingleBattleEnvDetensorizer,
)
from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvTensorizer import (
    ActionType,
    PlaythroughStep,
)


class ReplayExplorer:
    def __init__(self, replay_path: str, output_manager: GameOutputManager):
        """
        Initialize the replay explorer with a path to a replay file.

        Args:
            replay_path: Path to the replay file
            output_manager: Output manager for formatted text output
        """
        self.replay_path = replay_path
        self.output = output_manager

        # Add PlaythroughStep to safe globals for loading
        torch.serialization.add_safe_globals([PlaythroughStep, ActionType])

        # Load the playthrough data
        loaded_data = torch.load(replay_path, weights_only=False)

        # Determine data format (PlaythroughStep objects or just state tuples)
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

        # Create detensorizer for accurate state reconstruction
        self.detensorizer = SingleBattleEnvDetensorizer()

        self.output.print(
            f"Loaded replay with {self.total_steps} steps from {replay_path}"
        )
        if self.is_simple_tuple_format:
            self.output.print(
                "NOTE: This is a simple state-only replay without action information"
            )

    def print_state_summary(self, step_idx: int, decoded_state: Dict[str, Any]) -> None:
        """Print a summary of the state for one step."""
        # Player info
        player = decoded_state["player"]
        self.output.print_player_info(player["hp"], player["max_hp"], player["energy"])

        # Enemies info
        for i, enemy in enumerate(decoded_state["enemies"]):
            self.output.print_opponent_info(i + 1, enemy["hp"], enemy["max_hp"])
            if enemy.get("intent"):
                self.output.print_opponent_intent(
                    enemy["intent"]["name"], enemy["intent"]["amount"]
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
                self.output.print_player_action(
                    action_type, card_name, card_idx, target_idx
                )
            else:
                self.output.print_player_action(action_type, None, card_idx, target_idx)
        elif action_type == "END_TURN":
            self.output.print_player_action(action_type)
        else:
            # Regular NO_OP
            self.output.print_player_action(action_type)

    def print_full_replay(self) -> None:
        """Print the entire replay in a turn-by-turn format."""
        self.output.print_separator()
        self.output.print_header("FULL REPLAY")

        # For full PlaythroughStep format with action information, use the detailed display
        current_turn = 1

        # Track opponent actions to ensure they are properly displayed
        opponent_actions_by_turn = {}

        # First pass: identify opponent actions that follow END_TURN
        for step_idx in range(len(self.playthrough_data) - 1):
            current_step = self.playthrough_data[step_idx]
            next_step = self.playthrough_data[step_idx + 1]

            # Check if this is an end turn followed by a potential opponent action state
            if (
                self.detensorizer.is_end_turn_state(current_step)
                and next_step.action_type == ActionType.NO_OP
            ):
                # Decode the next state and check for opponent action
                next_state = self.detensorizer.decode_state(next_step)
                opponent_action = self.detensorizer.decode_opponent_action(next_state)

                if opponent_action:
                    # Store by turn number for later display
                    opponent_actions_by_turn[current_turn] = opponent_action

            # Track turn transitions
            if step_idx > 0 and self.detensorizer.is_end_turn_state(
                self.playthrough_data[step_idx - 1]
            ):
                if next_step.action_type == ActionType.NO_OP and step_idx + 2 < len(
                    self.playthrough_data
                ):
                    # This is a turn transition (END_TURN -> NO_OP -> next turn)
                    current_turn += 1

        # Reset for the actual printing
        current_turn = 1
        is_turn_start = True

        # Create a mapping to track which steps represent end of player turns
        end_turn_steps = set()
        opponent_action_steps = set()

        # Identify all end turn and opponent action steps
        for step_idx in range(len(self.playthrough_data)):
            step = self.playthrough_data[step_idx]
            if step.action_type == ActionType.END_TURN:
                end_turn_steps.add(step_idx)
            elif (
                step.action_type == ActionType.NO_OP
                and step_idx > 0
                and self.playthrough_data[step_idx - 1].action_type
                == ActionType.END_TURN
            ):
                opponent_action_steps.add(step_idx)

        current_turn = 1
        for step_idx in range(len(self.playthrough_data)):
            step = self.playthrough_data[step_idx]
            decoded_state = self.detensorizer.decode_state(step)

            # Check if we're starting a new turn
            if is_turn_start:
                self.output.print_separator()
                self.output.print_subheader(f"Playing turn {current_turn}")
                self.output.print_separator()

                # Print detailed state information
                self.output.print_separator()
                self.output.print("Current State:")

                # Player info
                player = decoded_state["player"]
                self.output.print_player_info(
                    player["hp"], player["max_hp"], player["energy"]
                )

                # Print player hand
                if player["hand"]:
                    self.output.print("Player Hand:")
                    for i, card in enumerate(player["hand"]):
                        self.output.print_card(i, card["name"], card.get("cost", 1))

                # Print discard pile if not empty
                if player["discard_pile"]:
                    self.output.print("Discard Pile:")
                    for i, card in enumerate(player["discard_pile"]):
                        self.output.print_card(i, card["name"], card.get("cost", 1))

                # Print player statuses
                if player["statuses"]:
                    self.output.print("Player Statuses:")
                    for status_name, amount in player["statuses"].items():
                        self.output.print_status(status_name, amount)

                # Print opponent information
                for enemy_idx, enemy in enumerate(decoded_state.get("enemies", [])):
                    self.output.print(
                        f"Opponent {enemy_idx+1} HP: {enemy['hp']}/{enemy['max_hp']}"
                    )

                    # Print opponent statuses
                    if enemy.get("statuses"):
                        self.output.print("Opponent Statuses:")
                        for status_name, amount in enemy["statuses"].items():
                            self.output.print_status(status_name, amount)

                    # Print opponent intent
                    if enemy.get("intent"):
                        self.output.print("Opponent action:")
                        self.output.print(
                            f"  Intent: {enemy['intent']['name']} with amount {enemy['intent']['amount']}"
                        )

                self.output.print_separator()
                is_turn_start = False

            # Determine if this is a special step
            is_opponent_action = step_idx in opponent_action_steps

            # Skip printing opponent action steps here - we'll handle them after end turn
            if is_opponent_action:
                continue

            # Print action based on type
            if step.action_type == ActionType.PLAY_CARD:
                self.output.print("  Action: PLAY_CARD")

                # After playing a card, print the updated state
                self.output.print_separator()
                self.output.print("Updated State after playing card:")
                player = decoded_state["player"]
                self.output.print_player_info(
                    player["hp"], player["max_hp"], player["energy"]
                )

                # Print updated player hand
                if player["hand"]:
                    self.output.print("Player Hand:")
                    for i, card in enumerate(player["hand"]):
                        self.output.print_card(i, card["name"], card.get("cost", 1))

                # Print updated opponent information
                for enemy_idx, enemy in enumerate(decoded_state.get("enemies", [])):
                    self.output.print(
                        f"Opponent {enemy_idx+1} HP: {enemy['hp']}/{enemy['max_hp']}"
                    )

                self.output.print_separator()
            elif step.action_type == ActionType.END_TURN:
                self.output.print("  Action: End Turn")

                # Print opponent action after the end turn
                if current_turn in opponent_actions_by_turn:
                    opponent_action = opponent_actions_by_turn[current_turn]
                    self.output.print(
                        f"Opponent {opponent_action['type']} action: {opponent_action['action']} with amount {opponent_action['amount']}."
                    )

                # The next non-opponent-action step will be a new turn
                is_turn_start = True
                current_turn += 1
            elif step.action_type == ActionType.NO_OP:
                # Only print NO_OP if it's a meaningful operation
                if not is_opponent_action:
                    self.output.print("  Action: No Operation")

        self.output.print_separator()
        self.output.print_header("END OF REPLAY")


def main() -> None:
    """Main entry point for the replay explorer."""
    parser = argparse.ArgumentParser(
        description="Explore a saved tensor replay as text."
    )
    parser.add_argument("replay_path", type=str, help="Path to the replay file")
    parser.add_argument(
        "--log-file", "-l", type=str, default=None, help="Path to log file (optional)"
    )
    args = parser.parse_args()

    # Check if the replay file exists
    if not os.path.exists(args.replay_path):
        print(f"Error: Replay file '{args.replay_path}' not found.")
        return

    # Initialize the output manager
    output = GameOutputManager(args.log_file)

    try:
        explorer = ReplayExplorer(args.replay_path, output)
        explorer.print_full_replay()
    except Exception as e:
        output.print(f"Error loading or exploring replay: {e}")

    # Close the output manager
    output.close()


if __name__ == "__main__":
    main()
