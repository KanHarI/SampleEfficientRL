import os
from typing import Optional, TextIO


class GameOutputManager:
    """
    A class to manage output formatting and logging for the Deckbuilder game.
    This provides consistent output formatting across different game components
    and supports writing to both console and log files.
    """

    def __init__(self, log_file_path: Optional[str] = None):
        """
        Initialize the output manager.

        Args:
            log_file_path: Optional path to a log file. If provided, all output will be written
                          to this file in addition to the console.
        """
        self.log_file: Optional[TextIO] = None
        if log_file_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            self.log_file = open(log_file_path, "w", encoding="utf-8")

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        if self.log_file:
            self.log_file.close()

    def print(self, message: str) -> None:
        """
        Print a message to console and log file if configured.

        Args:
            message: The message to print
        """
        print(message)
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()

    def print_separator(self) -> None:
        """Print a separator line."""
        self.print("\n" + "=" * 40 + "\n")

    def print_header(self, header: str) -> None:
        """
        Print a formatted header.

        Args:
            header: The header text
        """
        self.print_separator()
        self.print(header)
        self.print_separator()

    def print_subheader(self, subheader: str) -> None:
        """
        Print a formatted subheader.

        Args:
            subheader: The subheader text
        """
        self.print("\n" + "-" * 30)
        self.print(subheader)
        self.print("-" * 30 + "\n")

    def print_player_info(self, hp: int, max_hp: int, energy: int) -> None:
        """
        Print player health and energy information.

        Args:
            hp: Current player health
            max_hp: Maximum player health
            energy: Current player energy
        """
        self.print(f"Player HP: {hp}/{max_hp}, Energy: {energy}")

    def print_card(self, index: int, name: str, cost: int) -> None:
        """
        Print card information.

        Args:
            index: Card index
            name: Card name
            cost: Card cost
        """
        self.print(f"  [{index}] {name} (Cost: {cost})")

    def print_status(self, name: str, amount: int) -> None:
        """
        Print status information.

        Args:
            name: Status name
            amount: Status amount
        """
        self.print(f"  {name}: {amount}")

    def print_opponent_info(self, index: int, hp: int, max_hp: int) -> None:
        """
        Print opponent health information.

        Args:
            index: Opponent index
            hp: Current opponent health
            max_hp: Maximum opponent health
        """
        self.print(f"Opponent {index} HP: {hp}/{max_hp}")

    def print_opponent_action(
        self, opponent_type: str, action_type: str, amount: int
    ) -> None:
        """
        Print opponent action information.

        Args:
            opponent_type: Type of opponent
            action_type: Type of action
            amount: Action amount
        """
        self.print(
            f"Opponent {opponent_type} action: {action_type} with amount {amount}."
        )

    def print_opponent_intent(self, intent_name: str, intent_amount: int) -> None:
        """
        Print opponent intent information.

        Args:
            intent_name: Name of the intent
            intent_amount: Amount of the intent
        """
        self.print(f"  Intent: {intent_name} with amount {intent_amount}")

    def print_play_result(self, result: str) -> None:
        """
        Print the result of playing a card.

        Args:
            result: Result message
        """
        self.print(result)

    def print_game_over(self, message: str) -> None:
        """
        Print game over message.

        Args:
            message: Game over message
        """
        self.print_separator()
        self.print(message)
        self.print_separator()

    def print_turn_header(self, turn_number: int) -> None:
        """
        Print turn header.

        Args:
            turn_number: Turn number
        """
        self.print_separator()
        self.print(f"Turn {turn_number}")

    def print_player_action(
        self,
        action_type: str,
        card_name: Optional[str] = None,
        card_idx: Optional[int] = None,
        target_idx: Optional[int] = None,
    ) -> None:
        """
        Print player action information.

        Args:
            action_type: Type of action
            card_name: Name of the card being played
            card_idx: Index of the card being played
            target_idx: Index of the target
        """
        if action_type == "PLAY_CARD" and card_name and card_idx is not None:
            self.print(
                f"  Action: Play card {card_name} (index {card_idx}), targeting enemy {target_idx}"
            )
        elif action_type == "END_TURN":
            self.print("  Action: End Turn")
        elif action_type == "NO_OP":
            self.print("  Action: No Operation")
        else:
            self.print(f"  Action: {action_type}")

    def close(self) -> None:
        """Close the log file if open."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None
