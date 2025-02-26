import math
from typing import Any, Dict, List, Optional, Tuple, cast

import torch

from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvTensorizer import (
    BINARY_NUMBER_BITS,
    MAX_ENCODED_NUMBER,
    SUPPORTED_ENEMY_INTENT_TYPES,
    SUPPORTED_OPPONENT_TYPES,
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
        """Initialize the detensorizer with mappings for cards, statuses, intents, and opponent types."""
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
            
        # Create mapping for opponent types
        self.opponent_type_map: Dict[int, str] = {}
        for i, opponent_type in enumerate(cast(Any, SUPPORTED_OPPONENT_TYPES)):
            self.opponent_type_map[i + 1] = str(opponent_type.name)
            
        # Create mapping for action types
        self.action_type_map = {
            ActionType.PLAY_CARD.value: "PLAY_CARD",
            ActionType.END_TURN.value: "END_TURN",
            ActionType.NO_OP.value: "NO_OP",
        }
        
        # Create token type mapping with only valid enum values
        self.token_type_map = {
            TokenType.DRAW_PILE_CARD.value: TokenType.DRAW_PILE_CARD,
            TokenType.DISCARD_PILE_CARD.value: TokenType.DISCARD_PILE_CARD,
            TokenType.EXHAUST_PILE_CARD.value: TokenType.EXHAUST_PILE_CARD,
            TokenType.HAND_CARD.value: TokenType.HAND_CARD,
            TokenType.ENTITY_HP.value: TokenType.ENTITY_HP,
            TokenType.ENTITY_MAX_HP.value: TokenType.ENTITY_MAX_HP,
            TokenType.ENTITY_ENERGY.value: TokenType.ENTITY_ENERGY,
            TokenType.ENTITY_STATUS.value: TokenType.ENTITY_STATUS,
            TokenType.ENEMY_INTENT.value: TokenType.ENEMY_INTENT,
            TokenType.PLAYER_ACTION.value: TokenType.PLAYER_ACTION,
            TokenType.ENEMY_ACTION.value: TokenType.ENEMY_ACTION,
            TokenType.TURN_MARKER.value: TokenType.TURN_MARKER,
        }

    def _extract_numeric_value(self, encoded_number_tensor: torch.Tensor) -> int:
        """
        Extract a numeric value from the encoded binary representation.

        Args:
            encoded_number_tensor: The binary tensor encoding a number

        Returns:
            The decoded integer value
        """
        return round(encoded_number_tensor[BINARY_NUMBER_BITS].item() * MAX_ENCODED_NUMBER)


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
            opponent_type_indices,
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
                "exhaust_pile": [],
                "statuses": {},
            },
            "enemies": [],
            "action": {
                "type": step.action_type.name,
                "card_idx": step.card_idx,
                "target_idx": step.target_idx,
                "reward": step.reward,
                "turn_number": step.turn_number,
            },
            "action_history": [],
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
        exhaust_cards_seen = 0
        current_enemy_idx = 0
        
        # Extract turn number if present
        turn_number = step.turn_number  # Default to the one stored in the step

        for i in range(token_types.size(0)):
            # Skip zero tokens (padding)
            if token_types[i].item() == 0 and i > 0:
                continue

            token_type = int(token_types[i].item())
            
            # Handle turn marker
            if token_type == TokenType.TURN_MARKER.value:
                turn_number = self._extract_numeric_value(encoded_numbers[i])
                state["turn_number"] = turn_number
                continue

            # Handle draw pile cards
            if token_type == TokenType.DRAW_PILE_CARD.value:
                card_idx = int(card_uid_indices[i].item())
                if card_idx > 0:
                    card_name = self.card_uid_map.get(card_idx, f"Unknown({card_idx})")
                    card_info = {
                        "name": card_name,
                        "cost": 1,  # Default cost, we don't have actual cost in tensor
                    }
                    state["player"]["draw_pile"].append(card_info)
                    draw_cards_seen += 1
                    
            # Handle hand cards
            elif token_type == TokenType.HAND_CARD.value:
                card_idx = int(card_uid_indices[i].item())
                if card_idx > 0:
                    card_name = self.card_uid_map.get(card_idx, f"Unknown({card_idx})")
                    card_cost = self._extract_numeric_value(encoded_numbers[i])
                    state["player"]["hand"].append({
                        "name": card_name,
                        "cost": card_cost,
                    })
                    hand_cards_seen += 1

            # Handle discard pile cards
            elif token_type == TokenType.DISCARD_PILE_CARD.value:
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
                    
            # Handle exhaust pile cards
            elif token_type == TokenType.EXHAUST_PILE_CARD.value:
                card_idx = int(card_uid_indices[i].item())
                if card_idx > 0:
                    card_name = self.card_uid_map.get(card_idx, f"Unknown({card_idx})")
                    state["player"]["exhaust_pile"].append(
                        {
                            "name": card_name,
                            "cost": 1,  # Default cost, we don't have actual cost in tensor
                        }
                    )
                    exhaust_cards_seen += 1

            # Handle player HP
            elif token_type == TokenType.ENTITY_HP.value and player_position == 0:
                player_position += 1  # Mark as processed
                state["player"]["hp"] = self._extract_numeric_value(encoded_numbers[i])
                
            # Handle player max HP
            elif token_type == TokenType.ENTITY_MAX_HP.value and player_position == 1:
                player_position += 1  # Mark as processed
                state["player"]["max_hp"] = self._extract_numeric_value(encoded_numbers[i])
                
            # Handle player energy
            elif token_type == TokenType.ENTITY_ENERGY.value:
                state["player"]["energy"] = self._extract_numeric_value(encoded_numbers[i])
                
            # Handle player status
            elif token_type == TokenType.ENTITY_STATUS.value and current_enemy_idx == 0:
                status_idx = int(status_uid_indices[i].item())
                if status_idx > 0:
                    status_name = self.status_uid_map.get(status_idx, f"Unknown({status_idx})")
                    status_amount = self._extract_numeric_value(encoded_numbers[i])
                    state["player"]["statuses"][status_name] = status_amount
                
            # Handle enemy HP
            elif token_type == TokenType.ENTITY_HP.value:
                # Create a new enemy if this is the first HP value for this enemy
                if len(state["enemies"]) <= current_enemy_idx:
                    # Get opponent type if available
                    opponent_type = "Unknown"
                    if i < opponent_type_indices.size(0) and opponent_type_indices[i].item() > 0:
                        type_idx = int(opponent_type_indices[i].item())
                        opponent_type = self.opponent_type_map.get(type_idx, f"Unknown({type_idx})")
                    
                    state["enemies"].append({
                        "type": opponent_type,
                        "hp": 0,
                        "max_hp": 0,
                        "statuses": {},
                        "intents": []
                    })
                
                state["enemies"][current_enemy_idx]["hp"] = self._extract_numeric_value(encoded_numbers[i])
                
            # Handle enemy max HP
            elif token_type == TokenType.ENTITY_MAX_HP.value and current_enemy_idx > 0:
                # Ensure we have an enemy entry
                if len(state["enemies"]) > current_enemy_idx - 1:
                    state["enemies"][current_enemy_idx - 1]["max_hp"] = self._extract_numeric_value(encoded_numbers[i])
                
            # Handle enemy status
            elif token_type == TokenType.ENTITY_STATUS.value and current_enemy_idx > 0:
                status_idx = int(status_uid_indices[i].item())
                if status_idx > 0 and len(state["enemies"]) > current_enemy_idx - 1:
                    status_name = self.status_uid_map.get(status_idx, f"Unknown({status_idx})")
                    status_amount = self._extract_numeric_value(encoded_numbers[i])
                    state["enemies"][current_enemy_idx - 1]["statuses"][status_name] = status_amount
                
            # Handle enemy intent
            elif token_type == TokenType.ENEMY_INTENT.value and len(state["enemies"]) > 0:
                intent_idx = int(enemy_intent_indices[i].item())
                if intent_idx > 0:
                    intent_type = self.intent_type_map.get(intent_idx, f"Unknown({intent_idx})")
                    intent_amount = self._extract_numeric_value(encoded_numbers[i])
                    last_enemy_idx = len(state["enemies"]) - 1
                    state["enemies"][last_enemy_idx]["intents"].append({
                        "type": intent_type,
                        "value": intent_amount
                    })
                    # If it's the first intent, also set it as the main intent
                    if "intent" not in state["enemies"][last_enemy_idx]:
                        state["enemies"][last_enemy_idx]["intent"] = {
                            "name": intent_type,
                            "amount": intent_amount
                        }
                
            # Handle player action
            elif token_type == TokenType.PLAYER_ACTION.value:
                # Placeholder for future expansion
                pass
                
            # Handle enemy action
            elif token_type == TokenType.ENEMY_ACTION.value:
                # Record enemy action in action history
                if "action_history" not in state:
                    state["action_history"] = []
                
                action_type = "ENEMY_ACTION"
                move_type = "Unknown"
                enemy_idx = current_enemy_idx - 1 if current_enemy_idx > 0 else 0
                
                # Try to extract the move type from enemy intent indices
                if i < enemy_intent_indices.size(0) and enemy_intent_indices[i].item() > 0:
                    intent_idx = int(enemy_intent_indices[i].item())
                    move_type = self.intent_type_map.get(intent_idx, f"Unknown({intent_idx})")
                
                state["action_history"].append({
                    "type": action_type,
                    "enemy_idx": enemy_idx,
                    "move_type": move_type
                })

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
        Check if this step represents an end of turn action.

        Args:
            step: The step to check

        Returns:
            True if this is an end turn action
        """
        return step.action_type == ActionType.END_TURN

    def is_opponent_action_state(self, step: PlaythroughStep, prev_step: Optional[PlaythroughStep] = None) -> bool:
        """
        Check if this step likely represents an opponent action.
        This is a heuristic based on comparing with the previous state.

        Args:
            step: The current step
            prev_step: The previous step, if available

        Returns:
            True if this step likely represents an opponent action
        """
        # If no previous step, can't determine
        if prev_step is None:
            return False
            
        # If player has less HP than before, likely an opponent action happened
        player_hp_now = 0
        player_hp_prev = 0
        
        # Find player HP in current step
        token_types = step.state[0]
        encoded_numbers = step.state[5]
        
        # Assume player is the first entity
        for i in range(token_types.size(0)):
            if token_types[i].item() == TokenType.ENTITY_HP.value:
                player_hp_now = self._extract_numeric_value(encoded_numbers[i])
                break
                
        # Find player HP in previous step
        prev_token_types = prev_step.state[0]
        prev_encoded_numbers = prev_step.state[5]
        
        for i in range(prev_token_types.size(0)):
            if prev_token_types[i].item() == TokenType.ENTITY_HP.value:
                player_hp_prev = self._extract_numeric_value(prev_encoded_numbers[i])
                break
                
        # If player lost HP, likely an opponent action
        if player_hp_now < player_hp_prev:
            return True
            
        # Check for changes in player statuses that might indicate opponent action
        return False
        
    def get_turn_number(self, step: PlaythroughStep) -> int:
        """
        Extract the turn number from a step.

        Args:
            step: The step to extract turn number from

        Returns:
            The turn number, or 0 if not found
        """
        # Use the turn number stored in the step if available
        if hasattr(step, 'turn_number') and step.turn_number is not None:
            return step.turn_number
            
        # Otherwise, try to find a turn marker in the state
        token_types = step.state[0]
        encoded_numbers = step.state[5]
        
        for i in range(token_types.size(0)):
            if token_types[i].item() == TokenType.TURN_MARKER.value:
                return self._extract_numeric_value(encoded_numbers[i])
                
        return 0  # Default if no turn number is found
        
    def get_step_reward(self, step: PlaythroughStep) -> float:
        """
        Get the reward associated with a step.
        
        Args:
            step: The PlaythroughStep to process
            
        Returns:
            The reward value for this step
        """
        return step.reward if hasattr(step, "reward") else 0.0
        
    def decode_playthrough(self, steps: List[PlaythroughStep]) -> List[Dict[str, Any]]:
        """
        Decode a full playthrough into a list of state representations.
        
        Args:
            steps: List of PlaythroughSteps representing a game playthrough
            
        Returns:
            A list of decoded state dictionaries with actions and rewards
        """
        decoded_states = []
        
        for i, step in enumerate(steps):
            state = self.decode_state(step)
            
            # Add metadata
            state["turn_number"] = self.get_turn_number(step)
            state["reward"] = self.get_step_reward(step)
            state["step_index"] = i
            
            # Determine if this is an opponent action by checking the previous step
            prev_step = steps[i-1] if i > 0 else None
            state["is_opponent_action"] = self.is_opponent_action_state(step, prev_step)
            state["is_end_turn"] = self.is_end_turn_state(step)
            
            decoded_states.append(state)
            
        return decoded_states
