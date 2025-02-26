import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch

from SampleEfficientRL.Envs.Deckbuilder.Card import CardUIDs
from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
    DeckbuilderSingleBattleEnv,
)
from SampleEfficientRL.Envs.Deckbuilder.Opponent import NextMoveType, OpponentTypeUIDs
from SampleEfficientRL.Envs.Deckbuilder.Status import StatusUIDs

# Full support for all cards in the game
SUPPORTED_CARDS_UIDs: List[CardUIDs] = [
    # Curses
    CardUIDs.ASCENDERS_BANE,
    
    # Status cards
    CardUIDs.SLIMED,
    
    # Ironclad Starter Cards
    CardUIDs.BASH,
    CardUIDs.DEFEND,
    CardUIDs.STRIKE,
    
    # Common Cards
    CardUIDs.ANGER,
    CardUIDs.ARMAMENTS,
    CardUIDs.BODY_SLAM,
    CardUIDs.CLASH,
    CardUIDs.CLEAVE,
    CardUIDs.CLOTHESLINE,
    CardUIDs.FLEX,
    CardUIDs.HAVOC,
    CardUIDs.HEADBUTT,
    CardUIDs.HEAVY_BLADE,
    CardUIDs.IRON_WAVE,
    CardUIDs.PERFECTED_STRIKE,
    CardUIDs.POMMEL_STRIKE,
    CardUIDs.SHRUG_IT_OFF,
    CardUIDs.SWORD_BOOMERANG,
    CardUIDs.THUNDERCLAP,
    CardUIDs.TRUE_GRIT,
    CardUIDs.TWIN_STRIKE,
    CardUIDs.WARCRY,
    CardUIDs.WILD_STRIKE,
    
    # Uncommon Cards
    CardUIDs.BATTLE_TRANCE,
    CardUIDs.BLOOD_FOR_BLOOD,
    CardUIDs.BLOODLETTING,
    CardUIDs.BURNING_PACT,
    CardUIDs.CARNAGE,
    CardUIDs.COMBUST,
    CardUIDs.DARK_EMBRACE,
    CardUIDs.DISARM,
    CardUIDs.DROPKICK,
    CardUIDs.DUAL_WIELD,
    CardUIDs.ENTRENCH,
    CardUIDs.EVOLVE,
    CardUIDs.FEEL_NO_PAIN,
    CardUIDs.FIRE_BREATHING,
    CardUIDs.FLAME_BARRIER,
    CardUIDs.GHOSTLY_ARMOR,
    CardUIDs.HEMOKINESIS,
    CardUIDs.INFERNAL_BLADE,
    CardUIDs.INFLAME,
    CardUIDs.INTIMIDATE,
    CardUIDs.METALLICIZE,
    CardUIDs.POWER_THROUGH,
    CardUIDs.PUMMEL,
    CardUIDs.RAGE,
    CardUIDs.RAMPAGE,
    CardUIDs.RECKLESS_CHARGE,
    CardUIDs.RUPTURE,
    CardUIDs.SEARING_BLOW,
    CardUIDs.SECOND_WIND,
    CardUIDs.SEEING_RED,
    CardUIDs.SENTINEL,
    CardUIDs.SEVER_SOUL,
    CardUIDs.SHOCKWAVE,
    CardUIDs.SPOT_WEAKNESS,
    CardUIDs.UPPERCUT,
    CardUIDs.WHIRLWIND,
    
    # Rare Cards
    CardUIDs.BARRICADE,
    CardUIDs.BERSERK,
    CardUIDs.BLUDGEON,
    CardUIDs.BRUTALITY,
    CardUIDs.CORRUPTION,
    CardUIDs.DEMON_FORM,
    CardUIDs.DOUBLE_TAP,
    CardUIDs.EXHUME,
    CardUIDs.FEED,
    CardUIDs.FIEND_FIRE,
    CardUIDs.IMMOLATE,
    CardUIDs.IMPERVIOUS,
    CardUIDs.JUGGERNAUT,
    CardUIDs.LIMIT_BREAK,
    CardUIDs.OFFERING,
    CardUIDs.REAPER,
]

# Full support for all statuses in the game
SUPPORTED_STATUS_UIDs: List[StatusUIDs] = [
    StatusUIDs.VULNERABLE,
    StatusUIDs.WEAK,
    StatusUIDs.FRAIL,
    StatusUIDs.POISON,
    StatusUIDs.BLOCK,
    StatusUIDs.RITUAL,
    StatusUIDs.STRENGTH,
    StatusUIDs.HAND_DRAWER,
    StatusUIDs.ENERGY_USER,
]

# All enemy intent types
SUPPORTED_ENEMY_INTENT_TYPES: List[NextMoveType] = [
    NextMoveType.ATTACK,
    NextMoveType.RITUAL,
]

# Supported opponent types
SUPPORTED_OPPONENT_TYPES: List[OpponentTypeUIDs] = [
    OpponentTypeUIDs.CULTIST,
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
    DRAW_PILE_CARD = 0
    DISCARD_PILE_CARD = 1
    EXHAUST_PILE_CARD = 2
    HAND_CARD = 3
    ENTITY_HP = 4
    ENTITY_MAX_HP = 5
    ENTITY_ENERGY = 6
    ENTITY_STATUS = 7
    ENEMY_INTENT = 8
    PLAYER_ACTION = 9
    ENEMY_ACTION = 10
    TURN_MARKER = 11


class ActionType(Enum):
    PLAY_CARD = 0
    END_TURN = 1
    NO_OP = 2  # For states where no action is taken


NUM_MAX_ENEMIES = ENTITY_TYPE.ENEMY_6.value

MAX_ENCODED_NUMBER = 1023
BINARY_NUMBER_BITS = 10
SCALAR_NUMBER_DIMS = 1
LOG_NUMBER_DIMS = 1
SIGN_BITS = 1
NUMBER_ENCODING_DIMS = SIGN_BITS + BINARY_NUMBER_BITS + SCALAR_NUMBER_DIMS + LOG_NUMBER_DIMS


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
    include_turn_count: bool = True
    include_action_history: bool = True
    max_action_history: int = 10  # Maximum number of previous actions to include


@dataclass
class PlaythroughStep:
    """A single step in a playthrough, containing state and action information."""

    state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    action_type: ActionType
    card_idx: Optional[int] = None
    target_idx: Optional[int] = None
    reward: float = 0.0
    turn_number: int = 0


@dataclass
class GameStateCache:
    """Caches previous state information for action history recording."""
    
    last_action_type: Optional[ActionType] = None
    last_card_idx: Optional[int] = None
    last_target_idx: Optional[int] = None
    previous_actions: List[Tuple[ActionType, Optional[int], Optional[int]]] = None
    
    def __post_init__(self):
        if self.previous_actions is None:
            self.previous_actions = []
    
    def record_action(self, action_type: ActionType, card_idx: Optional[int] = None, target_idx: Optional[int] = None):
        """Record an action to the cache."""
        self.last_action_type = action_type
        self.last_card_idx = card_idx
        self.last_target_idx = target_idx
        self.previous_actions.append((action_type, card_idx, target_idx))


class SingleBattleEnvTensorizer:
    def __init__(self, config: SingleBattleEnvTensorizerConfig):
        self.config = config
        self.playthrough_steps: List[PlaythroughStep] = []
        self.state_cache = GameStateCache()
        # Map card UIDs to their indices for fast lookup
        self.card_uid_to_idx: Dict[CardUIDs, int] = {
            card_uid: idx + 1 for idx, card_uid in enumerate(SUPPORTED_CARDS_UIDs)
        }
        # Map status UIDs to their indices for fast lookup
        self.status_uid_to_idx: Dict[StatusUIDs, int] = {
            status_uid: idx + 1 for idx, status_uid in enumerate(SUPPORTED_STATUS_UIDs)
        }
        # Map enemy intent types to their indices for fast lookup
        self.enemy_intent_to_idx: Dict[NextMoveType, int] = {
            intent_type: idx + 1 for idx, intent_type in enumerate(SUPPORTED_ENEMY_INTENT_TYPES)
        }
        # Map opponent types to their indices for fast lookup
        self.opponent_type_to_idx: Dict[OpponentTypeUIDs, int] = {
            opponent_type: idx + 1 for idx, opponent_type in enumerate(SUPPORTED_OPPONENT_TYPES)
        }

    def _encode_number(self, num: int) -> torch.Tensor:
        """
        Encodes a number into a multi-dimensional representation:
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
        num = max(num, -MAX_ENCODED_NUMBER)
        # Initialize tensor for the encoded number
        encoded = torch.zeros(NUMBER_ENCODING_DIMS, dtype=torch.float)

        encoded[0] = 1.0 if num >= 0 else -1.0
        # Binary bits (10 bits)
        for i in range(BINARY_NUMBER_BITS):
            if num & (1 << i):
                encoded[i + SIGN_BITS] = 1.0

        # Scalar value (normalized to [0, 1])
        encoded[BINARY_NUMBER_BITS + SIGN_BITS] = float(num) / MAX_ENCODED_NUMBER

        # Log value
        if num > 0:
            encoded[BINARY_NUMBER_BITS + SCALAR_NUMBER_DIMS + SIGN_BITS] = math.log(float(num))
        else:
            encoded[BINARY_NUMBER_BITS + SCALAR_NUMBER_DIMS + SIGN_BITS] = -1.0

        return encoded

    # Return tensors tuple:
    # Token types,
    # Card uid indices,
    # Status uid indices,
    # Enemy intent type indices,
    # Opponent type indices,
    # Encoded numbers,
    def tensorize(
        self, state: DeckbuilderSingleBattleEnv
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            - opponent_type_indices: The indices of opponent types in the context.
            - encoded_numbers: The encoded numerical values in the context.

        Raises:
            ValueError: If the state representation exceeds the configured context size.
        """
        # Initialize tensors with zeros
        token_types = torch.zeros(self.config.context_size, dtype=torch.long)
        card_uid_indices = torch.zeros(self.config.context_size, dtype=torch.long)
        status_uid_indices = torch.zeros(self.config.context_size, dtype=torch.long)
        enemy_intent_indices = torch.zeros(self.config.context_size, dtype=torch.long)
        opponent_type_indices = torch.zeros(self.config.context_size, dtype=torch.long)
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

        # Encode turn number if configured
        if self.config.include_turn_count:
            check_context_size()
            token_types[position] = TokenType.TURN_MARKER.value
            encoded_numbers[position] = self._encode_number(state.num_turn)
            position += 1

        # Encode player's draw pile
        player = state.player
        if player is None:
            raise ValueError("Player is not set")

        for card in player.draw_pile:
            check_context_size()

            token_types[position] = TokenType.DRAW_PILE_CARD.value
            card_uid_indices[position] = self.card_uid_to_idx.get(card.card_uid, 0)
            encoded_numbers[position] = self._encode_number(card.cost)
            position += 1

        # Encode player's discard pile
        for card in player.discard_pile:
            check_context_size()

            token_types[position] = TokenType.DISCARD_PILE_CARD.value
            card_uid_indices[position] = self.card_uid_to_idx.get(card.card_uid, 0)
            encoded_numbers[position] = self._encode_number(card.cost)
            position += 1

        # Encode player's exhaust pile (if available)
        if hasattr(player, 'exhaust_pile'):
            for card in player.exhaust_pile:
                check_context_size()

                token_types[position] = TokenType.EXHAUST_PILE_CARD.value
                card_uid_indices[position] = self.card_uid_to_idx.get(card.card_uid, 0)
                encoded_numbers[position] = self._encode_number(card.cost)
                position += 1

        # Encode player's hand
        for card_idx, card in enumerate(player.hand):
            check_context_size()

            token_types[position] = TokenType.HAND_CARD.value
            card_uid_indices[position] = self.card_uid_to_idx.get(card.card_uid, 0)
            # Store the card index as a number so we can reference it when playing cards
            encoded_numbers[position] = self._encode_number(card.cost)
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

            token_types[position] = TokenType.ENTITY_STATUS.value
            status_uid_indices[position] = self.status_uid_to_idx.get(status_uid, 0)
            encoded_numbers[position] = self._encode_number(amount)
            position += 1

        # Encode action history if configured
        if self.config.include_action_history and self.state_cache.previous_actions:
            # Limit to max_action_history most recent actions
            recent_actions = self.state_cache.previous_actions[-self.config.max_action_history:]
            
            for action_type, card_idx, target_idx in recent_actions:
                check_context_size()
                
                token_types[position] = TokenType.PLAYER_ACTION.value
                if action_type == ActionType.PLAY_CARD and card_idx is not None:
                    # If we played a card, record which card it was
                    if 0 <= card_idx < len(player.hand):
                        card = player.hand[card_idx]
                        card_uid_indices[position] = self.card_uid_to_idx.get(card.card_uid, 0)
                # Store the action type, card index, and target index
                action_value = action_type.value
                encoded_numbers[position] = self._encode_number(action_value)
                position += 1

        # Encode enemies
        if state.opponents is None:
            raise ValueError("Opponents not set")

        for enemy_idx, enemy in enumerate(state.opponents):
            if enemy.current_health > 0:  # Check if enemy is alive using current_health
                # Enemy type
                check_context_size()
                opponent_type_indices[position] = self.opponent_type_to_idx.get(enemy.opponent_type_uid, 0)
                position += 1
                
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
                    enemy_intent_indices[position] = self.enemy_intent_to_idx.get(enemy.next_move.move_type, 0)
                    if enemy.next_move.amount is not None:
                        encoded_numbers[position] = self._encode_number(enemy.next_move.amount)
                    position += 1

                # Enemy statuses
                for status_uid, (status, amount) in enemy.get_active_statuses().items():
                    check_context_size()

                    token_types[position] = TokenType.ENTITY_STATUS.value
                    status_uid_indices[position] = self.status_uid_to_idx.get(status_uid, 0)
                    encoded_numbers[position] = self._encode_number(amount)
                    position += 1

        return (
            token_types,
            card_uid_indices,
            status_uid_indices,
            enemy_intent_indices,
            opponent_type_indices,
            encoded_numbers,
        )

    def record_action(
        self,
        state_tensor: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        action_type: ActionType,
        card_idx: Optional[int] = None,
        target_idx: Optional[int] = None,
        reward: float = 0.0,
        turn_number: int = 0,
    ) -> None:
        """
        Record an action taken in the given state.

        Args:
            state_tensor: The tensor representation of the state before the action.
            action_type: The type of action taken.
            card_idx: The index of the card played (if applicable).
            target_idx: The index of the target for the card (if applicable).
            reward: The reward received for taking this action.
            turn_number: The current turn number.
        """
        if self.config.mode != TensorizerMode.RECORD:
            return

        # Update state cache with this action
        self.state_cache.record_action(action_type, card_idx, target_idx)

        step = PlaythroughStep(
            state=state_tensor,
            action_type=action_type,
            card_idx=card_idx,
            target_idx=target_idx,
            reward=reward,
            turn_number=turn_number,
        )
        self.playthrough_steps.append(step)

    def get_playthrough_data(self) -> List[PlaythroughStep]:
        """Get the recorded playthrough data."""
        return self.playthrough_steps

    def clear_playthrough_data(self) -> None:
        """Clear the recorded playthrough data."""
        self.playthrough_steps = []
        self.state_cache = GameStateCache()

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
            turn_number=state.num_turn,
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
            state_tensor=state_tensor, 
            action_type=ActionType.END_TURN, 
            reward=reward,
            turn_number=state.num_turn,
        )
        
    def record_enemy_action(
        self, 
        state: DeckbuilderSingleBattleEnv,
        enemy_idx: int,
        move_type: NextMoveType,
        amount: Optional[int] = None,
        reward: float = 0.0,
    ) -> None:
        """
        Record an enemy action.
        
        Args:
            state: The current game state.
            enemy_idx: The index of the enemy taking the action.
            move_type: The type of move the enemy is making.
            amount: The amount associated with the move (e.g., attack damage).
            reward: The reward received for this action.
        """
        if self.config.mode != TensorizerMode.RECORD:
            return
            
        state_tensor = self.tensorize(state)
        # We store enemy actions in the state cache but don't create playthrough steps for them
        # This is because we focus on the player's decision points for RL
        # But tracking enemy actions helps with understanding the game state
        self.state_cache.record_action(
            ActionType.NO_OP,  # Use NO_OP as a placeholder for enemy actions
            card_idx=enemy_idx,  # Store enemy index here
            target_idx=self.enemy_intent_to_idx.get(move_type, 0)  # Store move type in target_idx
        )

    def save_playthrough(self, filename: str) -> None:
        """
        Save the recorded playthrough data to a file.

        Args:
            filename: The path to save the data to.
        """
        torch.save(self.playthrough_steps, filename)
        
    def load_playthrough(self, filename: str) -> None:
        """
        Load playthrough data from a file.
        
        Args:
            filename: The path to load the data from.
        """
        self.playthrough_steps = torch.load(filename)
