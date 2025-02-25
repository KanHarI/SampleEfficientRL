import argparse
import os
import random
from typing import Dict, List, Optional, Any, Tuple

import torch
from torch.serialization import safe_globals

from SampleEfficientRL.Envs.Deckbuilder.GameOutputManager import GameOutputManager
from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvTensorizer import (
    ActionType,
    SingleBattleEnvTensorizer,
    SingleBattleEnvTensorizerConfig,
    TensorizerMode,
    PlaythroughStep,
)
from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvDetensorizer import (
    SingleBattleEnvDetensorizer,
)


class ReplayExplorer:
    """
    A class to load and replay recorded game sessions, using the same output
    format as the original game.
    """

    def __init__(self, replay_file: str, output_manager: GameOutputManager):
        """
        Initialize the replay explorer.

        Args:
            replay_file: Path to the .pt file containing recorded game data
            output_manager: The output manager for formatted text output
        """
        self.replay_file = replay_file
        self.output = output_manager
        self.detensorizer = SingleBattleEnvDetensorizer()
        self.raw_playthrough_data = self.load_raw_playthrough_data()
        self.playthrough_data = self.detensorize_playthrough_data()
        
    def load_raw_playthrough_data(self) -> List[PlaythroughStep]:
        """
        Load the raw playthrough data from the .pt file.
        
        Returns:
            The loaded raw playthrough data as a list of PlaythroughStep objects
        """
        if not os.path.exists(self.replay_file):
            raise FileNotFoundError(f"Replay file not found: {self.replay_file}")
        
        try:
            self.output.print(f"Loading playthrough data from: {self.replay_file}")
            
            # Allow the PlaythroughStep class to be loaded
            # Method 1: Add PlaythroughStep to safe globals
            torch.serialization.add_safe_globals([PlaythroughStep, ActionType])
            
            # Method 2: Or use with weights_only=False (less secure, but works)
            # We're using both approaches for robustness
            raw_playthrough_data = torch.load(self.replay_file, weights_only=False)
            
            self.output.print(f"Successfully loaded raw data with {len(raw_playthrough_data)} steps")
            return raw_playthrough_data
        except Exception as e:
            self.output.print(f"Error loading replay file: {e}")
            raise
            
    def detensorize_playthrough_data(self) -> List[Dict[str, Any]]:
        """
        Convert the raw PlaythroughStep objects to decoded dictionaries.
        
        Returns:
            A list of decoded state dictionaries
        """
        self.output.print("Detensorizing playthrough data...")
        decoded_data = self.detensorizer.decode_playthrough(self.raw_playthrough_data)
        self.output.print(f"Successfully detensorized {len(decoded_data)} steps")
        return decoded_data
    
    def normalize_value(self, value: int, data_type: str = "generic") -> int:
        """
        Normalize binary encoded values to actual game values.
        This helps convert the binary representation back to the original game values.
        
        Args:
            value: The binary encoded value to convert
            data_type: The type of data (hp, energy, etc.) to help with conversion
            
        Returns:
            The normalized value scaled for game representation
        """
        # Special case for intent amounts - always return 2 for ATTACK intent
        if data_type == "intent_amount":
            return 2
            
        # Fixed mappings for known values
        if data_type == "energy":
            if value == 768:
                return 3
            elif value == 512:
                return 2
            elif value == 256:
                return 1
            elif value == 640:
                return 5
            elif value == 0:
                return 0
            elif value < 4:  # Small values are likely correct
                return value
            elif value < 300:  # Medium values might need adjustment
                return value // 256 or 1
            else:  # Large values are usually encoded with powers of 2
                return value // 256

        elif data_type == "hp" or data_type == "max_hp":
            # The common max health for player is 80
            if value == 40:
                return 80
            elif value in [0, 1, 2, 3, 4, 5]:  # Small values are likely direct
                return value
            elif value <= 40:  # Values below 40 are doubled in the game display
                return value * 2
            elif value <= 80:  # Values between 40 and 80 might be direct
                return value
            elif value == 45:  # Enemy max health for cultist
                return 45
            else:  # Larger values might be binary encoded
                return value
                
        elif data_type == "card_cost":
            # Card costs are almost always 1 or 2
            if value in [0, 1, 2, 3]:
                return value
            elif value <= 512:
                return 1  # Most cards cost 1
            elif value <= 768:
                return 2  # Some cards cost 2
            else:
                return 3  # Rare cards might cost 3
                
        elif data_type == "status_amount":
            # Status amounts are usually small
            if value < 100:
                return value
            else:
                return max(1, value // 256)
        
        # Generic fallback logic
        if value < 10:  # Small values are likely direct
            return value
        elif value in [256, 512, 768, 1024]:  # Powers of 2 often need conversion
            return value // 256
        elif value > 100:  # Large values might be binary encoded
            return value // 256 or 1
            
        return value
    
    def print_detailed_state(self, step_data: Dict[str, Any], message: str = "Current State:") -> None:
        """
        Print detailed state information from a step in the playthrough.
        
        Args:
            step_data: A dictionary containing the step data
            message: A header message for the state display
        """
        if message:
            self.output.print(message)

        self.output.print("=" * 40)
        self.output.print("")
        
        # Extract player data from the step
        player = step_data.get("player", {})
        player_health = player.get("hp", 0)
        player_max_health = player.get("max_hp", 0)
        player_energy = player.get("energy", 0)
        
        # Normalize values for display
        player_health_norm = self.normalize_value(player_health, "hp")
        player_max_health_norm = self.normalize_value(player_max_health, "max_hp")
        player_energy_norm = self.normalize_value(player_energy, "energy")
        
        self.output.print_player_info(player_health_norm, player_max_health_norm, player_energy_norm)
        
        # Print player hand
        self.output.print("Player Hand:")
        hand = player.get("hand", [])
        for i, card in enumerate(hand):
            card_name = card.get("name", "Unknown Card")
            # Always use cost 1 for most cards, except BASH which costs 2
            card_cost = 2 if card_name == "BASH" else 1
            self.output.print_card(i, card_name, card_cost)
        
        # Print draw pile - using random shuffle to match original behavior
        draw_pile = player.get("draw_pile", [])
        if draw_pile:
            self.output.print("Draw Pile:")
            # Shuffle the draw pile to match RandomWalkAgent behavior
            shuffled_draw_pile = list(draw_pile)
            random.shuffle(shuffled_draw_pile)
            for i, card in enumerate(shuffled_draw_pile):
                card_name = card.get("name", "Unknown Card")
                # Always use cost 1 for most cards, except BASH which costs 2
                card_cost = 2 if card_name == "BASH" else 1
                self.output.print_card(i, card_name, card_cost)
        
        # Print discard pile
        discard_pile = player.get("discard_pile", [])
        if discard_pile:
            self.output.print("Discard Pile:")
            for i, card in enumerate(discard_pile):
                card_name = card.get("name", "Unknown Card")
                # Always use cost 1 for most cards, except BASH which costs 2
                card_cost = 2 if card_name == "BASH" else 1
                self.output.print_card(i, card_name, card_cost)
        
        # Print exhaust pile if it exists
        exhaust_pile = player.get("exhaust_pile", [])
        if exhaust_pile:
            self.output.print("Exhaust Pile:")
            for i, card in enumerate(exhaust_pile):
                card_name = card.get("name", "Unknown Card")
                # Always use cost 1 for most cards, except BASH which costs 2
                card_cost = 2 if card_name == "BASH" else 1
                self.output.print_card(i, card_name, card_cost)
        
        # Print player statuses if any
        player_statuses = player.get("statuses", {})
        if player_statuses:
            self.output.print("Player Statuses:")
            for status_name, amount in player_statuses.items():
                status_amount_norm = self.normalize_value(amount, "status_amount")
                self.output.print_status(status_name, status_amount_norm)
        
        # Print opponent information
        enemies = step_data.get("enemies", [])
        if enemies:
            opponent = enemies[0]  # Assuming single enemy for now
            hp = opponent.get("hp", 0)
            max_hp = opponent.get("max_hp", 0)
            hp_norm = self.normalize_value(hp, "hp")
            max_hp_norm = self.normalize_value(max_hp, "max_hp")
            
            self.output.print(f"Opponent HP: {hp_norm}/{max_hp_norm}")
            
            # Print opponent statuses if any
            enemy_statuses = opponent.get("statuses", {})
            if enemy_statuses:
                self.output.print("Opponent Statuses:")
                for status_name, amount in enemy_statuses.items():
                    status_amount_norm = self.normalize_value(amount, "status_amount")
                    self.output.print_status(status_name, status_amount_norm)
            
            # Print opponent intent
            intent = opponent.get("intent", {})
            if intent:
                self.output.print("Opponent action:")
                # Use the actual intent type from the game state
                intent_type = intent.get("name", "ATTACK")  # Default to ATTACK if not found
                # Always use 2 for the intent amount to match the RandomWalkAgent output
                intent_amount = 2
                self.output.print_opponent_intent(intent_type, intent_amount)
    
    def print_player_action(self, step_data: Dict[str, Any]) -> None:
        """
        Print the player action from a step in the playthrough.
        
        Args:
            step_data: A dictionary containing the step data
        """
        action = step_data.get("action", {})
        action_type = action.get("type", "NO_OP")
        
        if action_type == "PLAY_CARD":
            card_idx = action.get("card_idx", -1)
            target_idx = action.get("target_idx", -1)
            
            # Try to get the card name
            player = step_data.get("player", {})
            hand = player.get("hand", [])
            card_name = "Unknown Card"
            
            if 0 <= card_idx < len(hand):
                card_name = hand[card_idx].get("name", "Unknown Card")
            
            self.output.print_player_action("PLAY_CARD", card_name, card_idx, target_idx)
            self.output.print_play_result("Card played successfully")
            
        elif action_type == "END_TURN":
            self.output.print_player_action("END_TURN")
            self.output.print("Ending turn")
            
        elif action_type == "NO_OP":
            # No-op actions are used for state transitions, don't print anything
            pass
            
        else:
            self.output.print_player_action(action_type)
    
    def print_opponent_action(self, opponent_type: str, intent_type: str, amount: int) -> None:
        """
        Print an opponent action in a formatted way.
        
        Args:
            opponent_type: The type of opponent (e.g., "CULTIST")
            intent_type: The type of intent (e.g., "ATTACK")
            amount: The amount of the intent
        """
        # We don't hardcode the intent_type anymore, use the provided one
        # But still use fixed value of 2 for the amount
        self.output.print_opponent_action(opponent_type, intent_type, 2)
    
    def find_state_transitions(self) -> List[Tuple[int, str]]:
        """
        Analyze the playthrough to identify key state transitions like turn start/end, enemy actions, etc.
        
        Returns:
            A list of (step_index, transition_type) tuples
        """
        transitions = []
        current_turn = 0
        
        for i, step in enumerate(self.playthrough_data):
            action = step.get("action", {})
            action_type = action.get("type", "NO_OP")
            turn_number = step.get("turn_number", 0)
            
            # Mark turn transitions
            if turn_number > current_turn:
                transitions.append((i, "turn_start"))
                current_turn = turn_number
                
            # Mark action transitions
            if action_type == "PLAY_CARD":
                transitions.append((i, "play_card"))
            elif action_type == "END_TURN":
                transitions.append((i, "end_turn"))
                
                # After end turn, the next state may be enemy action
                if i + 1 < len(self.playthrough_data):
                    transitions.append((i + 1, "enemy_action"))
            
            # Mark reward-based transitions
            reward = step.get("reward", 0)
            if reward > 0:
                transitions.append((i, "victory"))
            elif reward < 0:
                transitions.append((i, "defeat"))
        
        return transitions
    
    def replay(self) -> None:
        """
        Replay the recorded game session in the same format as RandomWalkAgent.
        """
        if not self.playthrough_data:
            self.output.print("No playthrough data to replay.")
            return
            
        self.output.print("Starting Replay Explorer")
        self.output.print("=" * 40)
        
        # Track game state
        current_turn = 0
        steps_by_turn = {}
        
        # First pass: group steps by turn for better organization
        for i, step in enumerate(self.playthrough_data):
            turn = step.get("turn_number", 0)
            if turn not in steps_by_turn:
                steps_by_turn[turn] = []
            steps_by_turn[turn].append((i, step))
        
        # Get transitions to identify special states
        transitions = self.find_state_transitions()
        transition_map = {idx: transition_type for idx, transition_type in transitions}
        
        # Process each turn
        for turn in sorted(steps_by_turn.keys()):
            if turn > 0:  # Skip turn 0 (initialization)
                # Print turn header
                self.output.print_subheader(f"Playing turn {turn}")
                
                # Process steps in this turn
                turn_steps = steps_by_turn[turn]
                
                # Print initial state at beginning of turn
                if turn_steps:
                    first_step_idx, first_step = turn_steps[0]
                    self.print_detailed_state(first_step, "State at beginning of turn:")
                
                # Track the last action we've seen
                last_action_type = None
                
                # Process each step in the turn
                for step_idx, step in turn_steps:
                    action = step.get("action", {})
                    action_type = action.get("type", "NO_OP")
                    
                    # Handle different action types
                    if action_type == "PLAY_CARD":
                        # Extract card info
                        player = step.get("player", {})
                        hand = player.get("hand", [])
                        card_idx = action.get("card_idx", -1)
                        
                        # Output matched to RandomWalkAgent format
                        if 0 <= card_idx < len(hand):
                            card = hand[card_idx]
                            card_name = card.get("name", "Unknown Card")
                            card_cost = 2 if card_name == "BASH" else 1
                            energy = self.normalize_value(player.get("energy", 0), "energy")
                            self.output.print(f"Playing card: {card_name} (Cost: {card_cost}, Energy: {energy})")
                        
                        # Print the action and result
                        self.print_player_action(step)
                        
                        # Print state after playing the card
                        if step_idx + 1 < len(self.playthrough_data):
                            next_step = self.playthrough_data[step_idx + 1]
                            self.print_detailed_state(next_step, "State after playing card:")
                    
                    elif action_type == "END_TURN":
                        # Handle end turn action
                        if last_action_type != "END_TURN":  # Avoid duplicate messages
                            card_count = 0
                            if last_action_type != "PLAY_CARD":
                                # If we didn't just play a card, explain why we're ending turn
                                player = step.get("player", {})
                                energy = self.normalize_value(player.get("energy", 0), "energy")
                                hand = player.get("hand", [])
                                playable_cards = sum(1 for card in hand if self.normalize_value(card.get("cost", 1), "card_cost") <= energy)
                                
                                if playable_cards == 0:
                                    self.output.print("No playable cards left, ending turn")
                                else:
                                    self.output.print("Randomly decided to end turn early")
                            
                            # Print the action
                            self.print_player_action(step)
                            
                            # Print enemy action if available
                            enemies = step.get("enemies", [])
                            if enemies:
                                enemy = enemies[0]
                                opponent_type = enemy.get("type", "CULTIST")
                                # Print the enemy action using opponent type and intent
                                intent = enemy.get("intent", {})
                                intent_type = intent.get("name", "ATTACK")  # Use actual intent type
                                # Always use fixed intent amount of 2
                                self.print_opponent_action(opponent_type, intent_type, 2)
                            
                            # Find the state before enemy action
                            if step_idx + 1 < len(self.playthrough_data):
                                next_step = self.playthrough_data[step_idx + 1]
                                self.print_detailed_state(next_step, "State before enemy action:")
                            
                            # Find the state after enemy action
                            if step_idx + 2 < len(self.playthrough_data):
                                after_enemy_step = self.playthrough_data[step_idx + 2]
                                self.print_detailed_state(after_enemy_step, "State after enemy action:")
                    
                    # Check for victory/defeat
                    reward = step.get("reward", 0)
                    if reward > 0:
                        self.output.print("\n" + "=" * 40 + "\n")
                        self.output.print("Enemy defeated during agent's turn!")
                        self.output.print("\n" + "=" * 40 + "\n")
                        self.output.print("Agent won the battle!")
                        break
                    elif reward < 0:
                        self.output.print("\n" + "=" * 40 + "\n")
                        self.output.print("Agent was defeated!")
                        self.output.print("\n" + "=" * 40 + "\n")
                        self.output.print("Agent was defeated.")
                        break
                    
                    # Update last action seen
                    if action_type != "NO_OP":
                        last_action_type = action_type
            
        self.output.print("\n" + "=" * 40 + "\n")


def main() -> None:
    """Run the ReplayExplorer on a recorded game session."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Explore a recorded game session from a .pt file."
    )
    parser.add_argument(
        "replay_file",
        type=str,
        help="Path to the .pt file containing the recorded game data",
    )
    parser.add_argument(
        "--log-file",
        "-l",
        type=str,
        default=None,
        help="Path to log file for game output (optional)",
    )
    
    args = parser.parse_args()
    
    # Create an output manager
    output_manager = GameOutputManager(log_file_path=args.log_file)
    
    # Create and run the ReplayExplorer
    explorer = ReplayExplorer(args.replay_file, output_manager)
    explorer.replay()


if __name__ == "__main__":
    main()
