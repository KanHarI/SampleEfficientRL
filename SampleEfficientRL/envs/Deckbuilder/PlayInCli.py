import argparse
import os
import random
import sys

from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
    DeckbuilderSingleBattleEnv,
)
from SampleEfficientRL.Envs.Deckbuilder.GameOutputManager import GameOutputManager
from SampleEfficientRL.Envs.Deckbuilder.IroncladStarterVsCultist import (
    IroncladStarterVsCultist,
)
from SampleEfficientRL.Envs.Deckbuilder.Player import PlayCardResult
from SampleEfficientRL.Envs.Deckbuilder.Status import StatusesOrder


def print_state(env: DeckbuilderSingleBattleEnv, output: GameOutputManager) -> None:
    player = env.player
    if player is None:
        raise ValueError("Player is not set")
    opponent = env.opponents[0] if env.opponents and len(env.opponents) > 0 else None
    if opponent is None:
        raise ValueError("Opponent is not set")

    output.print_separator()
    output.print("Current State:")
    output.print_player_info(player.current_health, player.max_health, player.energy)

    output.print("Player Hand:")
    for i, card in enumerate(player.hand):
        # Try to get card properties, fallback to defaults
        card_uid = card.card_uid
        card_cost = card.cost
        output.print_card(i, card_uid.name, card_cost)

    draw_pile = player.draw_pile.copy()
    random.shuffle(draw_pile)
    output.print("Draw Pile:")
    for i, card in enumerate(draw_pile):
        card_uid = card.card_uid
        card_cost = card.cost
        output.print_card(i, card_uid.name, card_cost)

    discard_pile = player.discard_pile.copy()
    output.print("Discard Pile:")
    for i, card in enumerate(discard_pile):
        card_uid = card.card_uid
        card_cost = card.cost
        output.print_card(i, card_uid.name, card_cost)

    output.print("Player Statuses:")
    for status_uid in StatusesOrder:
        if status_uid in player.get_active_statuses():
            status, amount = player.get_active_statuses()[status_uid]
            output.print_status(status_uid.name, amount)

    output.print(f"Opponent HP: {opponent.current_health}/{opponent.max_health}")

    output.print("Opponent Statuses:")
    for status_uid in StatusesOrder:
        if status_uid in opponent.get_active_statuses():
            status, amount = opponent.get_active_statuses()[status_uid]
            output.print_status(status_uid.name, amount)

    output.print("Opponent action:")
    opponent_action = opponent.next_move
    if opponent_action is None:
        output.print("No action")
    else:
        amount = 0
        if opponent_action.amount is not None:
            amount = opponent_action.amount
        output.print_opponent_action(
            opponent.opponent_type_uid.name, opponent_action.move_type.name, amount
        )
    output.print_separator()


def player_turn(env: DeckbuilderSingleBattleEnv, output: GameOutputManager) -> str:
    # Start player's turn events (draw hand, set energy from statuses)
    env.start_turn()
    player = env.player
    if player is None:
        raise ValueError("Player is not set")

    while True:
        print_state(env, output)

        user_input = input(
            "Enter card index to play or type 'end' or 'e' to finish turn, 'quit' or 'q' to quit: "
        ).strip()
        if user_input.lower() in ["end", "e"]:
            os.system("cls" if os.name == "nt" else "clear")
            print("\n")
            break
        if user_input.lower() in ["quit", "q", "exit"]:
            output.print("Exiting game.")
            sys.exit(0)
        try:
            index = int(user_input)
        except ValueError:
            os.system("cls" if os.name == "nt" else "clear")
            print("\n")
            output.print("Invalid input. Please enter a valid card index or 'end'.")
            continue

        os.system("cls" if os.name == "nt" else "clear")
        print("\n")
        if index < 0 or index >= len(player.hand):
            output.print("Invalid card index. Try again.")
            continue

        card = player.hand[index]
        if player.energy < card.cost:
            output.print(
                f"Not enough energy to play this card. Card cost is {card.cost} and you have {player.energy} energy."
            )
            continue

        card_uid = card.card_uid
        output.print(f"Playing card: {card_uid.name}.")

        # All cards target the first enemy (hack for now)
        result = env.play_card_from_hand(index, 0)
        if result == PlayCardResult.CARD_NOT_FOUND:
            output.print_play_result("Card not found in hand")
        if result == PlayCardResult.NOT_ENOUGH_ENERGY:
            output.print_play_result("Not enough energy to play this card")
        if result == PlayCardResult.CARD_PLAYED_SUCCESSFULLY:
            output.print_play_result("Card played successfully")

        # Process any events (like enemy death)
        events = env.emit_events()
        for event in events:
            if event.name == "WIN_BATTLE":
                output.print("Enemy defeated during your turn!")
                return "win"
            if event.name == "PLAYER_DEATH":
                output.print("You have died!")
                return "lose"

    return "continue"


def main() -> None:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Play the Ironclad vs Cultist CLI game"
    )
    parser.add_argument(
        "--log-file",
        "-l",
        type=str,
        default=None,
        help="Path to log file for game output (optional)",
    )
    args = parser.parse_args()

    # Initialize output manager with optional log file
    output = GameOutputManager(args.log_file)

    os.system("cls" if os.name == "nt" else "clear")
    print("\n")

    output.print_header("Starting Ironclad vs Cultist CLI Game")
    game = IroncladStarterVsCultist()
    player = game.player

    turn = 1
    while True:
        output.print_turn_header(turn)
        result = player_turn(game, output)
        if result == "win":
            output.print_game_over("Congratulations! You won the battle!")
            break
        if result == "lose" or getattr(player, "health", 1) <= 0:
            output.print_game_over("You have been defeated. Game Over.")
            break

        if game.opponents is None:
            raise ValueError("Opponents are not set")
        for opponent in game.opponents:
            next_move = opponent.next_move
            if next_move is None:
                raise ValueError("Opponent next move is not set")

            amount = 0
            if next_move.amount is not None:
                amount = next_move.amount

            output.print_opponent_action(
                opponent.opponent_type_uid.name, next_move.move_type.name, amount
            )

        game.end_turn()

        if player is None:
            raise ValueError("Player is not set")

        if player.current_health <= 0:
            output.print_game_over(
                "You have been defeated after the enemy's turn. Game Over."
            )
            break
        turn += 1

    # Close the output manager to ensure log file is properly closed
    output.close()


if __name__ == "__main__":
    main()
