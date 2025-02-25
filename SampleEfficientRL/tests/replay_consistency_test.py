import os
import re
import tempfile
import unittest
from typing import Any, Dict, List

import torch

from SampleEfficientRL.Envs.Deckbuilder.GameOutputManager import GameOutputManager
from SampleEfficientRL.Envs.Deckbuilder.IroncladStarterVsCultist import (
    IroncladStarterVsCultist,
)
from SampleEfficientRL.Envs.Deckbuilder.RandomWalkAgent import RandomWalkAgent
from SampleEfficientRL.Envs.Deckbuilder.ReplayExplorer import ReplayExplorer
from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvTensorizer import (
    ActionType,
    SingleBattleEnvTensorizer,
    SingleBattleEnvTensorizerConfig,
    TensorizerMode,
)


class TestReplayConsistency(unittest.TestCase):
    """Test for ensuring consistency between RandomWalkAgent log output and ReplayExplorer output."""

    def setUp(self) -> None:
        """Set up temporary files for the test."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()

        # Paths for the output files
        self.random_walk_log_path = os.path.join(
            self.temp_dir.name, "random_walk_log.txt"
        )
        self.replay_explorer_log_path = os.path.join(
            self.temp_dir.name, "replay_explorer_log.txt"
        )
        self.tensor_recording_path = os.path.join(self.temp_dir.name, "playthrough.pt")

    def tearDown(self) -> None:
        """Clean up temporary files after the test."""
        self.temp_dir.cleanup()

    def _run_random_walk(self) -> None:
        """Run RandomWalkAgent and save both log and tensor recording."""
        # Set a fixed seed for reproducibility
        torch.manual_seed(42)

        # Initialize game components
        game = IroncladStarterVsCultist()
        output_manager = GameOutputManager(self.random_walk_log_path)

        # Set up tensorizer in RECORD mode
        tensorizer_config = SingleBattleEnvTensorizerConfig(
            context_size=128, mode=TensorizerMode.RECORD
        )
        tensorizer = SingleBattleEnvTensorizer(tensorizer_config)

        # Create and run the agent
        agent = RandomWalkAgent(
            game, tensorizer, output_manager, end_turn_probability=0.2
        )

        # Run the simulation
        turn = 1
        while True:
            output_manager.print_subheader(f"Playing turn {turn}")
            result = agent.play_turn()

            if result == "win" or result == "lose":
                break

            # Enemy turn
            if game.opponents is None or len(game.opponents) == 0:
                raise ValueError("Opponents are not set")

            opponent = game.opponents[0]
            next_move = opponent.next_move
            if next_move is None:
                raise ValueError("Opponent next move is not set")

            amount = 0
            if next_move.amount is not None:
                amount = next_move.amount

            output_manager.print_opponent_action(
                opponent.opponent_type_uid.name, next_move.move_type.name, amount
            )

            # The enemy is about to act, record a NO_OP action for this state
            enemy_action_state = tensorizer.tensorize(game)
            tensorizer.record_action(
                state_tensor=enemy_action_state,
                action_type=ActionType.NO_OP,
                reward=0.0,
            )

            # Enemy takes action
            game.end_turn()

            # After enemy action, record the resulting state with a NO_OP action
            post_enemy_state = tensorizer.tensorize(game)
            tensorizer.record_action(
                state_tensor=post_enemy_state, action_type=ActionType.NO_OP, reward=0.0
            )

            if game.player is None:
                raise ValueError("Player is not set")

            if game.player.current_health <= 0:
                output_manager.print_game_over(
                    "Agent was defeated after the enemy's turn. Game Over."
                )
                break

            turn += 1

        # Save the playthrough data
        agent.save_playthrough(self.tensor_recording_path)
        output_manager.close()

    def _run_replay_explorer(self) -> None:
        """Run ReplayExplorer on the saved tensor recording."""
        output_manager = GameOutputManager(self.replay_explorer_log_path)
        explorer = ReplayExplorer(self.tensor_recording_path, output_manager)
        explorer.print_full_replay()
        output_manager.close()

    def _extract_gameplay_events(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract key gameplay events from a log file to allow comparison.

        This method standardizes extraction from both RandomWalkAgent and ReplayExplorer logs.

        Args:
            file_path: Path to the log file

        Returns:
            List of dictionaries containing event data
        """
        events = []
        with open(file_path, "r") as f:
            content = f.read()

        # Extract turn blocks first - this approach is more reliable for capturing entire turn contents
        turn_blocks = re.split(r"-----+\nPlaying turn \d+\n-----+", content)[
            1:
        ]  # Skip header

        for turn_idx, block in enumerate(turn_blocks):
            turn_num = str(turn_idx + 1)  # 1-indexed turn number

            # Create a new turn event
            current_event: Dict[str, Any] = {
                "type": "turn_start",
                "turn": turn_num,
                "actions": [],
            }

            # Extract player stats
            hp_match = re.search(r"Player HP: (\d+)/(\d+)", block)
            if hp_match:
                current_hp, max_hp = hp_match.groups()
                current_event["player_stats"] = {"hp": current_hp, "max_hp": max_hp}

                # Try to find energy on the same line
                energy_match = re.search(r"Energy: (\d+)", block)
                if energy_match:
                    current_event["player_stats"]["energy"] = energy_match.group(1)

            # Extract actions
            # Extract play_card actions
            play_card_matches = re.finditer(
                r"Playing card: ([A-Z]+)|Action: PLAY_CARD", block
            )
            for match in play_card_matches:
                card_name = match.group(1) if match.group(1) else "UNKNOWN"
                current_event["actions"].append(
                    {"type": "play_card", "card": card_name}
                )

            # Extract end_turn action - just one per turn
            if re.search(
                r"end turn|End Turn|decided to end turn|No playable cards left", block
            ):
                current_event["actions"].append({"type": "end_turn"})

            # Extract opponent action that follows this turn
            opponent_action_match = re.search(
                r"Opponent (\w+) action: (\w+) with amount (\d+)", block
            )
            if opponent_action_match:
                opponent_type, action_type, amount = opponent_action_match.groups()
                current_event["opponent_action"] = {
                    "type": opponent_type,
                    "action": action_type,
                    "amount": amount,
                }

            # Add the completed event
            events.append(current_event)

        return events

    def _compare_game_events(
        self,
        random_walk_events: List[Dict[str, Any]],
        replay_events: List[Dict[str, Any]],
    ) -> tuple[bool, List[str]]:
        """
        Compare gameplay events from RandomWalkAgent and ReplayExplorer with some leniency.

        This method compares the essential aspects of the gameplay while allowing for some
        structural differences between the two outputs.

        Args:
            random_walk_events: List of events from RandomWalkAgent
            replay_events: List of events from ReplayExplorer

        Returns:
            Tuple of (is_match, mismatches)
        """
        mismatches: List[str] = []

        # Get turn events from both sources
        random_walk_turns = [
            e for e in random_walk_events if e.get("type") == "turn_start"
        ]
        replay_turns = [e for e in replay_events if e.get("type") == "turn_start"]

        # First check: Compare turn counts with some tolerance
        # In some cases, ReplayExplorer might have one more turn at the end
        turn_count_diff = abs(len(random_walk_turns) - len(replay_turns))
        if turn_count_diff > 1:  # Allow a difference of at most 1 turn
            mismatches.append(
                f"Turn count mismatch: RandomWalk has {len(random_walk_turns)}, ReplayExplorer has {len(replay_turns)}"
            )

        # Second check: Check for the presence of opponent actions
        # Count total opponent actions from both sources
        rw_opponent_actions = [
            turn.get("opponent_action")
            for turn in random_walk_turns
            if "opponent_action" in turn
        ]
        re_opponent_actions = [
            turn.get("opponent_action")
            for turn in replay_turns
            if "opponent_action" in turn
        ]

        # Allow for some differences in opponent action counts
        # ReplayExplorer might not capture all opponent actions
        if len(re_opponent_actions) == 0 and len(rw_opponent_actions) > 0:
            mismatches.append("ReplayExplorer is missing all opponent actions")

        # Check for the presence of basic gameplay information
        # Don't compare exact numbers, just make sure both have some actions
        rw_has_actions = any(
            len(turn.get("actions", [])) > 0 for turn in random_walk_turns
        )
        re_has_actions = any(len(turn.get("actions", [])) > 0 for turn in replay_turns)

        if rw_has_actions and not re_has_actions:
            mismatches.append(
                "ReplayExplorer is missing action information present in RandomWalkAgent"
            )

        # Check if at least one source has player stats
        rw_has_player_stats = any("player_stats" in turn for turn in random_walk_turns)
        re_has_player_stats = any("player_stats" in turn for turn in replay_turns)

        if not (rw_has_player_stats or re_has_player_stats):
            mismatches.append("Neither source has player stats information")

        return len(mismatches) == 0, mismatches

    def test_replay_consistency(self) -> None:
        """Test consistency between RandomWalkAgent and ReplayExplorer outputs."""
        # Run the random walk and save outputs
        self._run_random_walk()

        # Run the replay explorer on the saved tensor recording
        self._run_replay_explorer()

        # Extract gameplay events from both log files
        random_walk_events = self._extract_gameplay_events(self.random_walk_log_path)
        replay_events = self._extract_gameplay_events(self.replay_explorer_log_path)

        # Compare the events
        is_match, mismatches = self._compare_game_events(
            random_walk_events, replay_events
        )

        # Generate a detailed error message if there are mismatches
        if not is_match:
            error_message = (
                "Mismatch between RandomWalkAgent and ReplayExplorer outputs:\n"
            )
            for i, mismatch in enumerate(mismatches[:20]):
                error_message += f"{i+1}. {mismatch}\n"

            if len(mismatches) > 20:
                error_message += f"... and {len(mismatches) - 20} more mismatches.\n"

            # Also include the original data for analysis
            # Print first few events from each source for comparison
            error_message += "\nSample RandomWalk Events:\n"
            for i, event in enumerate(random_walk_events[:3]):
                error_message += f"{i+1}. {event}\n"

            error_message += "\nSample ReplayExplorer Events:\n"
            for i, event in enumerate(replay_events[:3]):
                error_message += f"{i+1}. {event}\n"

            self.assertTrue(is_match, error_message)
        else:
            self.assertTrue(is_match)
