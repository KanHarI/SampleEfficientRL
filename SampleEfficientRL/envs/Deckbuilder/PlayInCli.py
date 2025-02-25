import os
import random
import sys
import argparse
from typing import Optional

from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
    DeckbuilderSingleBattleEnv,
)
from SampleEfficientRL.Envs.Deckbuilder.IroncladStarterVsCultist import (
    IroncladStarterVsCultist,
)
from SampleEfficientRL.Envs.Deckbuilder.Player import PlayCardResult
from SampleEfficientRL.Envs.Deckbuilder.Status import StatusesOrder
from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvTensorizer import (
    SingleBattleEnvTensorizer,
    SingleBattleEnvTensorizerConfig,
    TensorizerMode,
    ActionType,
)


def print_separator() -> None:
    print("\n" + "=" * 40 + "\n")


def print_state(env: DeckbuilderSingleBattleEnv) -> None:
    player = env.player
    if player is None:
        raise ValueError("Player is not set")
    opponent = env.opponents[0] if env.opponents and len(env.opponents) > 0 else None
    if opponent is None:
        raise ValueError("Opponent is not set")
    print_separator()
    print("Current State:")
    print(f"Player HP: {player.current_health}")
    print(f"Player Energy: {player.energy}")
    print("Player Hand:")
    for i, card in enumerate(player.hand):
        # Try to get card properties, fallback to defaults
        card_uid = card.card_uid
        card_cost = card.cost
        print(f"  [{i}] {card_uid.name} (Cost: {card_cost})")
    draw_pile = player.draw_pile.copy()
    random.shuffle(draw_pile)
    print("Draw Pile:")
    for i, card in enumerate(draw_pile):
        card_uid = card.card_uid
        card_cost = card.cost
        print(f"  [{i}] {card_uid.name} (Cost: {card_cost})")
    discard_pile = player.discard_pile.copy()
    print("Discard Pile:")
    for i, card in enumerate(discard_pile):
        card_uid = card.card_uid
        card_cost = card.cost
        print(f"  [{i}] {card_uid.name} (Cost: {card_cost})")
    print("Player Statuses:")
    for status_uid in StatusesOrder:
        if status_uid in player.get_active_statuses():
            status, amount = player.get_active_statuses()[status_uid]
            print(f"  {status_uid.name}: {amount}")
    print(f"Opponent HP: {opponent.current_health}")
    print("Opponent Statuses:")
    for status_uid in StatusesOrder:
        if status_uid in opponent.get_active_statuses():
            status, amount = opponent.get_active_statuses()[status_uid]
            print(f"  {status_uid.name}: {amount}")
    print("Opponent action:")
    opponent_action = opponent.next_move
    if opponent_action is None:
        print("No action")
    else:
        print(f"{opponent_action.move_type.name} with amount {opponent_action.amount}.")
    print_separator()


def player_turn(env: DeckbuilderSingleBattleEnv, tensorizer: Optional[SingleBattleEnvTensorizer] = None) -> str:
    # Start player's turn events (draw hand, set energy from statuses)
    env.start_turn()
    player = env.player
    if player is None:
        raise ValueError("Player is not set")

    while True:
        print_state(env)

        user_input = input(
            "Enter card index to play or type 'end' or 'e' to finish turn, 'quit' or 'q' to quit: "
        ).strip()
        if user_input.lower() in ["end", "e"]:
            os.system("cls" if os.name == "nt" else "clear")
            # Record end turn action if recording is enabled
            if tensorizer:
                tensorizer.record_end_turn(env)
            break
        if user_input.lower() in ["quit", "q", "exit"]:
            print("Exiting game.")
            sys.exit(0)
        try:
            index = int(user_input)
        except ValueError:
            os.system("cls" if os.name == "nt" else "clear")
            print("Invalid input. Please enter a valid card index or 'end'.")
            continue

        os.system("cls" if os.name == "nt" else "clear")
        if index < 0 or index >= len(player.hand):
            print("Invalid card index. Try again.")
            continue

        card = player.hand[index]
        if player.energy < card.cost:
            print(
                f"Not enough energy to play this card. Card cost is {card.cost} and you have {player.energy} energy."
            )
            continue

        card_uid = card.card_uid
        print(f"Playing card: {card_uid.name}.")

        # Record the state and action before playing the card if recording is enabled
        if tensorizer:
            tensorizer.record_play_card(env, index, 0)  # Target first enemy

        # All cards target the first enemy (hack for now)
        result = env.play_card_from_hand(index, 0)
        if result == PlayCardResult.CARD_NOT_FOUND:
            print("Card not found in hand")
        if result == PlayCardResult.NOT_ENOUGH_ENERGY:
            print("Not enough energy to play this card")
        if result == PlayCardResult.CARD_PLAYED_SUCCESSFULLY:
            print("Card played successfully")

        # Process any events (like enemy death)
        events = env.emit_events()
        for event in events:
            if event.name == "WIN_BATTLE":
                print("Enemy defeated during your turn!")
                return "win"
            if event.name == "PLAYER_DEATH":
                print("You have died!")
                return "lose"

    return "continue"


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Play Ironclad vs Cultist CLI Game with optional replay recording."
    )
    parser.add_argument(
        "--record",
        "-r",
        type=str,
        help="Record the playthrough and save to the specified file path.",
    )
    args = parser.parse_args()

    # Initialize tensorizer if recording is enabled
    tensorizer = None
    if args.record:
        print(f"Recording enabled. Playthrough will be saved to: {args.record}")
        os.makedirs(os.path.dirname(args.record), exist_ok=True)
        tensorizer_config = SingleBattleEnvTensorizerConfig(
            context_size=128, mode=TensorizerMode.RECORD
        )
        tensorizer = SingleBattleEnvTensorizer(tensorizer_config)

    os.system("cls" if os.name == "nt" else "clear")
    print_separator()
    print("Starting Ironclad vs Cultist CLI Game")
    game = IroncladStarterVsCultist()
    player = game.player

    turn = 1
    while True:
        print_separator()
        print(f"Turn {turn}")
        result = player_turn(game, tensorizer)
        if result == "win":
            print("Congratulations! You won the battle!")
            break
        if result == "lose" or getattr(player, "health", 1) <= 0:
            print("You have been defeated. Game Over.")
            break

        if game.opponents is None:
            raise ValueError("Opponents are not set")
        
        # Record state before enemy action if recording is enabled
        if tensorizer:
            enemy_action_state = tensorizer.tensorize(game)
            tensorizer.record_action(
                state_tensor=enemy_action_state, action_type=ActionType.NO_OP, reward=0.0
            )
            
        for opponent in game.opponents:
            next_move = opponent.next_move
            if next_move is None:
                raise ValueError("Opponent next move is not set")
            print(
                f"Opponent {opponent.opponent_type_uid.name} action: {next_move.move_type.name} with amount {next_move.amount}."
            )

        game.end_turn()
        
        # Record state after enemy action if recording is enabled
        if tensorizer:
            post_enemy_state = tensorizer.tensorize(game)
            tensorizer.record_action(
                state_tensor=post_enemy_state, action_type=ActionType.NO_OP, reward=0.0
            )

        if player is None:
            raise ValueError("Player is not set")

        if player.current_health <= 0:
            print("You have been defeated after the enemy's turn. Game Over.")
            break
        turn += 1
        
    print_separator()
    print("Game Over.")
    
    # Save playthrough data if recording was enabled
    if args.record and tensorizer:
        tensorizer.save_playthrough(args.record, notes="Human player CLI session")
        print(f"Saved playthrough data with {len(tensorizer.get_playthrough_data())} steps to {args.record}")


if __name__ == "__main__":
    main()
