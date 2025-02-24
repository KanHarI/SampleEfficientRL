import os
import argparse
import torch
from typing import List, Dict, Optional, Tuple, Any

from SampleEfficientRL.Envs.Deckbuilder.Card import CardUIDs
from SampleEfficientRL.Envs.Deckbuilder.Status import StatusUIDs
from SampleEfficientRL.Envs.Deckbuilder.Opponent import NextMoveType
from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvTensorizer import (
    PlaythroughStep,
    ActionType,
    TokenType,
    SUPPORTED_CARDS_UIDs,
    SUPPORTED_STATUS_UIDs,
    SUPPORTED_ENEMY_INTENT_TYPES,
    MAX_ENCODED_NUMBER,
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
        # Add PlaythroughStep to safe globals for loading
        torch.serialization.add_safe_globals([PlaythroughStep, ActionType])
        self.playthrough_data = torch.load(replay_path, weights_only=False)
        self.total_steps = len(self.playthrough_data)
        
        # Mapping dictionaries for readable output
        self.card_uid_map = {i+1: uid.name for i, uid in enumerate(SUPPORTED_CARDS_UIDs)}
        self.status_uid_map = {i+1: uid.name for i, uid in enumerate(SUPPORTED_STATUS_UIDs)}
        self.intent_type_map = {i+1: intent.name for i, intent in enumerate(SUPPORTED_ENEMY_INTENT_TYPES)}
        
        print(f"Loaded replay with {self.total_steps} steps from {replay_path}")
    
    def _decode_state(self, step: PlaythroughStep) -> Dict[str, Any]:
        """
        Decode a step's state tensors into a readable format.
        
        Args:
            step: The PlaythroughStep to decode
            
        Returns:
            A dictionary with decoded state information
        """
        token_types, card_uid_indices, status_uid_indices, enemy_intent_indices, encoded_numbers = step.state
        
        # Initialize state container
        state = {
            'player': {
                'hp': 0,
                'max_hp': 0,
                'energy': 0,
                'hand': [],
                'draw_pile': [],
                'discard_pile': [],
                'statuses': {}
            },
            'enemies': [],
            'action': {
                'type': step.action_type.name,
                'card_idx': step.card_idx,
                'target_idx': step.target_idx,
                'reward': step.reward
            }
        }
        
        # Process each token in the state
        current_enemy = None
        
        for i in range(token_types.size(0)):
            # Skip zero tokens (padding)
            if token_types[i].item() == 0 and i > 0:
                continue
                
            token_type = token_types[i].item()
            
            # Handle draw deck cards
            if token_type == TokenType.DRAW_DECK_CARD.value:
                card_idx = card_uid_indices[i].item()
                if card_idx > 0:
                    card_name = self.card_uid_map.get(card_idx, f"Unknown({card_idx})")
                    # First few cards are considered hand, if there are no specific tokens for hand
                    if len(state['player']['hand']) < 5 and len(state['player']['draw_pile']) == 0:
                        state['player']['hand'].append({
                            'name': card_name,
                            'cost': 1  # Default cost, we don't have actual cost in tensor
                        })
                    else:
                        state['player']['draw_pile'].append({
                            'name': card_name,
                            'cost': 1  # Default cost, we don't have actual cost in tensor
                        })
            
            # Handle discard pile cards
            elif token_type == TokenType.DISCARD_DECK_CARD.value:
                card_idx = card_uid_indices[i].item()
                if card_idx > 0:
                    card_name = self.card_uid_map.get(card_idx, f"Unknown({card_idx})")
                    state['player']['discard_pile'].append({
                        'name': card_name,
                        'cost': 1  # Default cost, we don't have actual cost in tensor
                    })
            
            # Handle entity HP
            elif token_type == TokenType.ENTITY_HP.value:
                hp_value = encoded_numbers[i].item()
                # If we don't have player HP yet, this is player HP
                if state['player']['hp'] == 0:
                    state['player']['hp'] = hp_value
                else:
                    # Create a new enemy if needed
                    if current_enemy is None:
                        current_enemy = {'hp': 0, 'max_hp': 0, 'intent': None, 'statuses': {}}
                    current_enemy['hp'] = hp_value
            
            # Handle entity max HP
            elif token_type == TokenType.ENTITY_MAX_HP.value:
                max_hp_value = encoded_numbers[i].item()
                # If we don't have player max HP yet, this is player max HP
                if state['player']['max_hp'] == 0:
                    state['player']['max_hp'] = max_hp_value
                elif current_enemy is not None:
                    current_enemy['max_hp'] = max_hp_value
                    # Add enemy to list if we have all required info
                    if current_enemy['hp'] > 0 and current_enemy['max_hp'] > 0:
                        state['enemies'].append(current_enemy)
                        current_enemy = None
            
            # Handle entity energy
            elif token_type == TokenType.ENTITY_ENERGY.value:
                state['player']['energy'] = encoded_numbers[i].item()
            
            # Handle entity status
            elif token_type == TokenType.ENTITY_STATUS.value:
                status_idx = status_uid_indices[i].item()
                status_amount = encoded_numbers[i].item()
                if status_idx > 0:
                    status_name = self.status_uid_map.get(status_idx, f"Unknown({status_idx})")
                    # If we don't have enemy yet or have added it to list, this is player status
                    if current_enemy is None or current_enemy in state['enemies']:
                        state['player']['statuses'][status_name] = status_amount
                    else:
                        current_enemy['statuses'][status_name] = status_amount
            
            # Handle enemy intent
            elif token_type == TokenType.ENEMY_INTENT.value:
                intent_idx = enemy_intent_indices[i].item()
                intent_amount = encoded_numbers[i].item()
                if intent_idx > 0 and current_enemy is not None:
                    intent_name = self.intent_type_map.get(intent_idx, f"Unknown({intent_idx})")
                    current_enemy['intent'] = {'name': intent_name, 'amount': intent_amount}
        
        # Make sure all enemies are added
        if current_enemy is not None and current_enemy not in state['enemies']:
            state['enemies'].append(current_enemy)
            
        return state
    
    def print_state_summary(self, step_idx: int, decoded_state: Dict[str, Any]) -> None:
        """Print a summary of the state for one step."""
        # Player info
        player = decoded_state['player']
        print(f"Player HP: {player['hp']}/{player['max_hp']}, Energy: {player['energy']}")
        
        # Enemies info
        for i, enemy in enumerate(decoded_state['enemies']):
            print(f"Opponent {i+1} HP: {enemy['hp']}/{enemy['max_hp']}")
            if enemy['intent']:
                print(f"  Intent: {enemy['intent']['name']} with amount {enemy['intent']['amount']}")
    
    def print_action_summary(self, step_idx: int, decoded_state: Dict[str, Any]) -> None:
        """Print a summary of the action for one step."""
        action = decoded_state['action']
        action_type = action['type']
        
        if action_type == 'PLAY_CARD' and action['card_idx'] is not None:
            card_idx = action['card_idx']
            target_idx = action['target_idx']
            if card_idx < len(decoded_state['player']['hand']):
                card_name = decoded_state['player']['hand'][card_idx]['name']
                print(f"  Action: Play card {card_name} (index {card_idx}), targeting enemy {target_idx}")
            else:
                print(f"  Action: Play card at index {card_idx}, targeting enemy {target_idx}")
        elif action_type == 'END_TURN':
            print("  Action: End Turn")
        elif action_type == 'NO_OP':
            print("  Action: No Operation")
    
    def print_full_replay(self) -> None:
        """Print the entire replay in a turn-by-turn format."""
        print_separator()
        print("FULL REPLAY")
        print_separator()
        
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
            if step_idx == 0 or (step_idx > 0 and (
                # Start of player turn (we just detected a new turn)
                (player_turn and not in_enemy_phase) or
                # Start of enemy phase
                (in_enemy_phase and step_idx > 0 and 
                 self.playthrough_data[step_idx-1].action_type == ActionType.END_TURN and 
                 action_type == ActionType.NO_OP)
            )):
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
                    if step_idx + 1 < self.total_steps and self.playthrough_data[step_idx + 1].action_type == ActionType.NO_OP:
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
    parser = argparse.ArgumentParser(description='Explore a saved tensor replay as text.')
    parser.add_argument('replay_path', type=str, help='Path to the replay file')
    args = parser.parse_args()
    
    # Check if the replay file exists
    if not os.path.exists(args.replay_path):
        print(f"Error: Replay file '{args.replay_path}' not found.")
        return
    
    try:
        explorer = ReplayExplorer(args.replay_path)
        explorer.print_full_replay()
    except Exception as e:
        print(f"Error loading or exploring replay: {e}")


if __name__ == "__main__":
    main() 