import json
import time
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

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
    NO_OP = 2
    ENEMY_ACTION = 3


class PlaythroughStep:
    """Represents a single step in a game playthrough."""

    def __init__(
        self,
        state: Dict[str, Any],
        action_type: ActionType,
        reward: float,
        turn: int,
        enemy_action: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.state = state
        self.action_type = action_type
        self.reward = reward
        self.turn = turn
        self.enemy_action = enemy_action


NUM_MAX_ENEMIES = ENTITY_TYPE.ENEMY_6.value


class TensorizerMode(Enum):
    TRAINING = auto()
    EVALUATION = auto()
    REPLAY = auto()
    RECORD = auto()  # Added for compatibility with RandomWalkAgent.py


class SingleBattleEnvTensorizerConfig:
    """Configuration options for the SingleBattleEnvTensorizer."""

    def __init__(
        self,
        mode: TensorizerMode = TensorizerMode.TRAINING,
        save_interval: int = 1000,
        record_all_actions: bool = True,
        context_size: int = 10,  # Added for compatibility with RandomWalkAgent.py
        include_turn_count: bool = True,  # Added for compatibility with RandomWalkAgent.py
        include_action_history: bool = True,  # Added for compatibility with RandomWalkAgent.py
    ) -> None:
        self.mode = mode
        self.save_interval = save_interval
        self.record_all_actions = record_all_actions
        self.context_size = context_size
        self.include_turn_count = include_turn_count
        self.include_action_history = include_action_history


# Number of dimensions needed for the binary encoding of numbers
NUMBER_ENCODING_DIMS = 12  # scalar + sign bit + 10 binary bits


class SingleBattleEnvTensorizer:
    """
    Converts DeckbuilderSingleBattleEnv states into tensor representations
    and logs game actions for later analysis or replay.
    """

    def __init__(
        self, config: Optional[SingleBattleEnvTensorizerConfig] = None
    ) -> None:
        self.tensor_records: List[Dict[str, Any]] = []
        self.current_turn: int = 0
        self.version: str = "1.0"  # Version for serialization compatibility
        self.config = config or SingleBattleEnvTensorizerConfig()

    def _convert_number_to_binary(self, num: int) -> Tuple[int, int, List[int]]:
        """
        Converts a number to the specified binary format:
        - Scalar value
        - Sign bit (0 for positive, 1 for negative)
        - 10 binary bits

        Returns:
            Tuple of (scalar_value, sign_bit, binary_list)
        """
        scalar_value = num
        sign_bit = 0 if num >= 0 else 1
        abs_value = abs(num)

        # Convert to 10-bit binary representation
        binary = []
        for i in range(9, -1, -1):
            bit = (abs_value >> i) & 1
            binary.append(bit)

        return (scalar_value, sign_bit, binary)

    def _encode_card(self, card_uid: CardUIDs) -> List[int]:
        """
        Encodes a card as a one-hot vector based on the supported cards.

        Returns:
            A one-hot encoded vector representing the card.
        """
        encoding = [0] * len(SUPPORTED_CARDS_UIDs)
        if card_uid in SUPPORTED_CARDS_UIDs:
            index = SUPPORTED_CARDS_UIDs.index(card_uid)
            encoding[index] = 1
        return encoding

    def _encode_status(self, status_uid: StatusUIDs, value: int) -> List[Any]:
        """
        Encodes a status and its value.

        Returns:
            A list containing the status one-hot encoding and binary value representation.
        """
        # One-hot encode the status type
        status_encoding = [0] * len(SUPPORTED_STATUS_UIDs)
        if status_uid in SUPPORTED_STATUS_UIDs:
            index = SUPPORTED_STATUS_UIDs.index(status_uid)
            status_encoding[index] = 1

        # Convert the value to binary representation
        value_binary = self._convert_number_to_binary(value)

        return status_encoding + [value_binary]

    def _encode_enemy_intent(
        self, intent_type: NextMoveType, value: Optional[int]
    ) -> List[Any]:
        """
        Encodes an enemy's next move intent.

        Returns:
            A list containing the intent one-hot encoding and binary value representation.
        """
        # One-hot encode the intent type
        intent_encoding = [0] * len(SUPPORTED_ENEMY_INTENT_TYPES)
        if intent_type in SUPPORTED_ENEMY_INTENT_TYPES:
            index = SUPPORTED_ENEMY_INTENT_TYPES.index(intent_type)
            intent_encoding[index] = 1

        # Convert the value to binary representation, using 0 if value is None
        value_binary = self._convert_number_to_binary(value if value is not None else 0)

        return intent_encoding + [value_binary]

    def tensorize(self, env: DeckbuilderSingleBattleEnv) -> Dict[str, Any]:
        """
        Converts the current game state into a tensor record.

        Args:
            env: The DeckbuilderSingleBattleEnv instance.

        Returns:
            A dictionary containing the tensorized state components and turn number.
        """
        state_components = []

        # Player HP
        if env.player is not None:
            player_hp = self._convert_number_to_binary(env.player.current_health)
            player_max_hp = self._convert_number_to_binary(env.player.max_health)
            player_energy = self._convert_number_to_binary(env.player.energy)
            state_components.append(
                [TokenType.ENTITY_HP.value, ENTITY_TYPE.PLAYER.value, player_hp]
            )
            state_components.append(
                [TokenType.ENTITY_MAX_HP.value, ENTITY_TYPE.PLAYER.value, player_max_hp]
            )
            state_components.append(
                [TokenType.ENTITY_ENERGY.value, ENTITY_TYPE.PLAYER.value, player_energy]
            )

            # Player statuses
            for status_uid, (status, value) in env.player.statuses.items():
                status_encoding = self._encode_status(status_uid, value)
                state_components.append(
                    [
                        TokenType.ENTITY_STATUS.value,
                        ENTITY_TYPE.PLAYER.value,
                        status_encoding,
                    ]
                )

        # Enemies information
        if env.opponents is not None:
            for enemy_idx, enemy in enumerate(env.opponents, 1):
                if enemy_idx > NUM_MAX_ENEMIES:
                    break

                # Enemy HP
                enemy_hp = self._convert_number_to_binary(enemy.current_health)
                enemy_max_hp = self._convert_number_to_binary(enemy.max_health)
                state_components.append(
                    [TokenType.ENTITY_HP.value, enemy_idx, enemy_hp]
                )
                state_components.append(
                    [TokenType.ENTITY_MAX_HP.value, enemy_idx, enemy_max_hp]
                )

                # Enemy statuses
                for status_uid, (status, value) in enemy.statuses.items():
                    status_encoding = self._encode_status(status_uid, value)
                    state_components.append(
                        [TokenType.ENTITY_STATUS.value, enemy_idx, status_encoding]
                    )

                # Enemy intent
                if enemy.next_move is not None:
                    intent_type = enemy.next_move.move_type
                    intent_value = enemy.next_move.amount
                    intent_encoding = self._encode_enemy_intent(
                        intent_type, intent_value
                    )
                    state_components.append(
                        [TokenType.ENEMY_INTENT.value, enemy_idx, intent_encoding]
                    )

        # Deck information
        if env.player is not None:
            # Draw pile
            for card in env.player.draw_pile:
                card_encoding = self._encode_card(card.card_uid)
                state_components.append(
                    [TokenType.DRAW_PILE_CARD.value, 0, card_encoding]
                )

            # Hand
            for card_idx, card in enumerate(env.player.hand):
                card_encoding = self._encode_card(card.card_uid)
                state_components.append(
                    [TokenType.HAND_CARD.value, card_idx, card_encoding]
                )

            # Discard pile
            for card in env.player.discard_pile:
                card_encoding = self._encode_card(card.card_uid)
                state_components.append(
                    [TokenType.DISCARD_PILE_CARD.value, 0, card_encoding]
                )

        # Turn marker
        turn_marker = self._convert_number_to_binary(self.current_turn)
        state_components.append([TokenType.TURN_MARKER.value, 0, turn_marker])

        # Convert to tensor - need to flatten or change the structure to make it compatible with torch.tensor
        # Method 1: Convert to dictionary representation
        state_dict = {
            "state_components": state_components,
            "turn": self.current_turn,
        }

        return state_dict

    def record_play_card(
        self,
        env: DeckbuilderSingleBattleEnv,
        card_idx: int,
        target_idx: int,
        reward: float,
    ) -> None:
        """
        Logs a tensorized record for a card play action.

        Args:
            env: The DeckbuilderSingleBattleEnv instance.
            card_idx: The index of the card in the player's hand.
            target_idx: The index of the target (0 for player, 1+ for enemies).
            reward: The reward received for this action.
        """
        state_dict = self.tensorize(env)

        # Create action tensor - we'll store this in the record directly through record_action
        # rather than keeping a separate local variable
        self.record_action(state_dict, ActionType.PLAY_CARD, reward, self.current_turn)

    def record_end_turn(self, env: DeckbuilderSingleBattleEnv, reward: float) -> None:
        """
        Logs a tensorized record for ending a turn.

        Args:
            env: The DeckbuilderSingleBattleEnv instance.
            reward: The reward received for this action.
        """
        state_dict = self.tensorize(env)

        # Record the action
        self.record_action(state_dict, ActionType.END_TURN, reward, self.current_turn)

        # Increment turn counter
        self.current_turn += 1

    def record_action(
        self,
        state_dict: Dict[str, Any],
        action_type: ActionType,
        reward: float,
        turn_number: int,
    ) -> None:
        """
        Records a generic action along with the tensorized state.

        Args:
            state_dict: The dictionary containing the tensorized state components and turn number.
            action_type: The type of action being performed.
            reward: The reward received for this action.
            turn_number: The current turn number.
        """
        # Create record
        record = {
            "state": state_dict,
            "action_type": action_type.value,
            "reward": reward,
            "turn": turn_number,
        }

        # Add to records
        self.tensor_records.append(record)

    def record_enemy_action(
        self,
        env: DeckbuilderSingleBattleEnv,
        enemy_idx: int,
        move_type: NextMoveType,
        amount: Optional[int],
        reward: float,
    ) -> None:
        """
        Logs enemy actions with the corresponding tensorized game state.

        Args:
            env: The DeckbuilderSingleBattleEnv instance.
            enemy_idx: The index of the enemy performing the action.
            move_type: The type of move the enemy is performing.
            amount: The amount/value associated with the move. Can be None for moves that don't have an amount.
            reward: The reward (typically negative for player) for this enemy action.
        """
        state_dict = self.tensorize(env)

        # Record the action - no need to keep enemy_action as a separate variable
        record = {
            "state": state_dict,
            "action_type": ActionType.ENEMY_ACTION.value,
            "enemy_idx": enemy_idx,
            "move_type": (
                move_type.name if hasattr(move_type, "name") else str(move_type)
            ),
            "amount": amount,
            "reward": reward,
            "turn": self.current_turn,
        }

        self.tensor_records.append(record)

    def save_playthrough(self, filename: str) -> None:
        """
        Serializes and saves the tensorized playthrough data to a binary file.

        Args:
            filename: The name of the file to save the data to.
        """
        # Create header
        header = {
            "version": self.version,
            "timestamp": time.time(),
            "num_records": len(self.tensor_records),
            "tensor_size": (
                len(self.tensor_records[0]["state"]["state_components"])
                if self.tensor_records
                else (0,)
            ),
            "supported_cards": [card.name for card in SUPPORTED_CARDS_UIDs],
            "supported_statuses": [status.name for status in SUPPORTED_STATUS_UIDs],
            "supported_enemy_intents": [
                intent.name for intent in SUPPORTED_ENEMY_INTENT_TYPES
            ],
        }

        # Save data
        with open(filename, "wb") as f:
            # Write header size first (as 4 bytes)
            header_bytes = json.dumps(header).encode("utf-8")
            header_size = len(header_bytes)
            f.write(header_size.to_bytes(4, byteorder="little"))

            # Write header
            f.write(header_bytes)

            # Write each tensor record
            for record in self.tensor_records:
                # Convert dictionary to JSON and then to bytes
                state_bytes = json.dumps(record["state"]).encode("utf-8")

                # Write metadata
                metadata = {
                    "action_type": record["action_type"],
                    "reward": record["reward"],
                    "turn": record["turn"],
                }

                # Add enemy-specific fields if this is an enemy action
                if record["action_type"] == ActionType.ENEMY_ACTION.value:
                    metadata["enemy_idx"] = record["enemy_idx"]
                    metadata["move_type"] = record["move_type"]
                    metadata["amount"] = record["amount"]

                metadata_bytes = json.dumps(metadata).encode("utf-8")

                # Write metadata size
                f.write(len(metadata_bytes).to_bytes(4, byteorder="little"))

                # Write metadata
                f.write(metadata_bytes)

                # Write tensor size
                f.write(len(state_bytes).to_bytes(8, byteorder="little"))

                # Write tensor data
                f.write(state_bytes)

    def get_playthrough_data(self) -> List[Any]:
        """
        Returns the in-memory tensor records for further processing or inspection.

        Returns:
            The list of tensor records.
        """
        return self.tensor_records
