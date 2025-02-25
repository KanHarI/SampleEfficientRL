import argparse
import os
import random
from typing import Any, List

from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
    DeckbuilderSingleBattleEnv,
)
from SampleEfficientRL.Envs.Deckbuilder.GameOutputManager import GameOutputManager
from SampleEfficientRL.Envs.Deckbuilder.IroncladStarterVsCultist import (
    IroncladStarterVsCultist,
)
from SampleEfficientRL.Envs.Deckbuilder.Player import PlayCardResult
from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvTensorizer import (
    ActionType,
    SingleBattleEnvTensorizer,
    SingleBattleEnvTensorizerConfig,
    TensorizerMode,
)


class RandomWalkAgent:
    def __init__(
        self,
        env: DeckbuilderSingleBattleEnv,
        tensorizer: SingleBattleEnvTensorizer,
        output_manager: GameOutputManager,
        end_turn_probability: float = 0.2,
    ):
        """
        Agent that makes random choices in the deckbuilder game and records the playthrough.

        Args:
            env: The game environment
            tensorizer: The tensorizer to convert game states to tensors
            output_manager: The output manager for formatted text output
            end_turn_probability: Probability of ending the turn prematurely
        """
        self.env = env
        self.tensorizer = tensorizer
        self.output = output_manager
        self.end_turn_probability = end_turn_probability
        self.playthrough_data: List[Any] = []  # Store tensor states here

    def record_state(self) -> None:
        """Record the current state as tensors and add to playthrough data."""
        tensor_state = self.tensorizer.tensorize(self.env)
        self.playthrough_data.append(tensor_state)

    def play_turn(self) -> str:
        """
        Play a turn with random card choices and random turn ending.

        Returns:
            A string indicating the result: 'win', 'lose', or 'continue'
        """
        self.env.start_turn()
        player = self.env.player
        if player is None:
            raise ValueError("Player is not set")

        # Record initial state at start of turn (no longer needed with tensorizer recording)
        # self.record_state()

        while True:
            # Randomly decide whether to end the turn prematurely
            if random.random() < self.end_turn_probability:
                self.output.print("Randomly decided to end turn early")
                # Record the end turn action
                self.tensorizer.record_end_turn(self.env)
                break

            # Get playable cards (cost <= current energy)
            playable_cards = [
                (i, card)
                for i, card in enumerate(player.hand)
                if card.cost <= player.energy
            ]

            if not playable_cards:
                self.output.print("No playable cards left, ending turn")
                # Record the end turn action
                self.tensorizer.record_end_turn(self.env)
                break

            # Randomly select a card to play
            card_index, card = random.choice(playable_cards)
            self.output.print(
                f"Playing card: {card.card_uid.name} (Cost: {card.cost}, Energy: {player.energy})"
            )

            # Record the state and action before playing the card
            self.tensorizer.record_play_card(self.env, card_index, 0)

            # All cards target the first enemy (hack as in the original code)
            result = self.env.play_card_from_hand(card_index, 0)

            if result == PlayCardResult.CARD_NOT_FOUND:
                self.output.print_play_result("Card not found in hand")
            elif result == PlayCardResult.NOT_ENOUGH_ENERGY:
                self.output.print_play_result("Not enough energy to play this card")
            elif result == PlayCardResult.CARD_PLAYED_SUCCESSFULLY:
                self.output.print_play_result("Card played successfully")

            # Record state after playing card (no longer needed with tensorizer recording)
            # self.record_state()

            # Check for game-ending events
            events = self.env.emit_events()
            for event in events:
                if event.name == "WIN_BATTLE":
                    self.output.print("Enemy defeated during agent's turn!")
                    return "win"
                if event.name == "PLAYER_DEATH":
                    self.output.print("Agent has died!")
                    return "lose"

        return "continue"

    def save_playthrough(self, filename: str) -> None:
        """Save the recorded playthrough data to a binary file."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Use the tensorizer's save_playthrough method instead
        self.tensorizer.save_playthrough(filename)
        self.output.print(
            f"Saved playthrough data with {len(self.tensorizer.get_playthrough_data())} steps to {filename}"
        )


def main() -> None:
    """Run the random walk agent on a game of Ironclad vs Cultist."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run a random walk agent through the deckbuilder game."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=os.path.join("playthrough_data", "random_walk_playthrough.pt"),
        help="Output file path to save the playthrough data (default: playthrough_data/random_walk_playthrough.pt)",
    )
    parser.add_argument(
        "--log-file",
        "-l",
        type=str,
        default=None,
        help="Path to log file for game output (optional)",
    )
    args = parser.parse_args()

    # Initialize the output manager
    output = GameOutputManager(args.log_file)

    output.print_header("Starting Random Walk Agent simulation")
    output.print(f"Will save playthrough data to: {args.output}")

    # Set up game
    game = IroncladStarterVsCultist()

    # Set up tensorizer in RECORD mode
    tensorizer_config = SingleBattleEnvTensorizerConfig(
        context_size=128, mode=TensorizerMode.RECORD
    )
    tensorizer = SingleBattleEnvTensorizer(tensorizer_config)

    # Create agent
    agent = RandomWalkAgent(game, tensorizer, output)

    # Play game
    turn = 1
    while True:
        output.print_subheader(f"Playing turn {turn}")
        result = agent.play_turn()

        if result == "win":
            output.print_game_over("Agent won the battle!")
            break
        if result == "lose":
            output.print_game_over("Agent was defeated.")
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

        output.print_opponent_action(
            opponent.opponent_type_uid.name, next_move.move_type.name, amount
        )

        # The enemy is about to act, record a NO_OP action for this state
        enemy_action_state = tensorizer.tensorize(game)
        tensorizer.record_action(
            state_tensor=enemy_action_state, action_type=ActionType.NO_OP, reward=0.0
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
            output.print_game_over(
                "Agent was defeated after the enemy's turn. Game Over."
            )
            break

        turn += 1

    # Save playthrough data
    agent.save_playthrough(args.output)

    # Close the output manager
    output.close()


if __name__ == "__main__":
    main()
