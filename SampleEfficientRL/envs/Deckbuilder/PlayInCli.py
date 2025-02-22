import random
import sys

from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
    DeckbuilderSingleBattleEnv,
)
from SampleEfficientRL.Envs.Deckbuilder.EnvAction import EntityDescriptor
from SampleEfficientRL.Envs.Deckbuilder.IroncladStarterVsCultist import (
    IroncladStarterVsCultist,
)
from SampleEfficientRL.Envs.Deckbuilder.Status import StatusesOrder


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
        card_name = card.card_uid
        card_cost = card.cost
        print(f"  [{i}] {card_name} (Cost: {card_cost})")
    draw_pile = player.draw_pile.copy()
    random.shuffle(draw_pile)
    print("Draw Pile:")
    for i, card in enumerate(draw_pile):
        card_name = card.card_uid
        card_cost = card.cost
        print(f"  [{i}] {card_name} (Cost: {card_cost})")
    discard_pile = player.discard_pile.copy()
    print("Discard Pile:")
    for i, card in enumerate(discard_pile):
        card_name = card.card_uid
        card_cost = card.cost
        print(f"  [{i}] {card_name} (Cost: {card_cost})")
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
    print_separator()


def player_turn(env: DeckbuilderSingleBattleEnv) -> str:
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
            break
        if user_input.lower() in ["quit", "q", "exit"]:
            print("Exiting game.")
            sys.exit(0)
        try:
            index = int(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid card index or 'end'.")
            continue

        if index < 0 or index >= len(player.hand):
            print("Invalid card index. Try again.")
            continue

        card = player.hand[index]
        if player.energy < card.cost:
            print(
                f"Not enough energy to play this card. Card cost is {card.cost} and you have {player.energy} energy."
            )
            continue

        card_name = card.card_uid
        print(f"Playing card: {card_name}.")

        # All cards target the first enemy (hack for now)
        player.play_card(index, 0)

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


def enemy_turn(env: DeckbuilderSingleBattleEnv, turn: int) -> None:
    print_separator()
    print("Enemy Turn")
    if env.opponents is None:
        raise ValueError("Opponents are not set")
    if len(env.opponents) == 0:
        raise ValueError("No opponents remaining")
    # Each opponent takes a turn. Assuming a single opponent.
    for idx, enemy in enumerate(env.opponents):
        try:
            move = enemy.select_move(turn)  # get enemy's move
        except Exception as e:
            print(f"Error selecting enemy move: {e}")
            continue
        move_type = getattr(move, "move_type", None)
        move_amount = getattr(move, "amount", None)
        print(
            f"Enemy {idx} uses move: {move_type.name if move_type else 'Unknown'} with amount {move_amount}."
        )
        enemy.perform_move(EntityDescriptor(is_player=True), move)
    print("End of Enemy Turn.")
    env.end_turn()


def main() -> None:
    print_separator()
    print("Starting Ironclad vs Cultist CLI Game")
    game = IroncladStarterVsCultist()
    player = game.player

    turn = 1
    while True:
        print_separator()
        print(f"Turn {turn}")
        result = player_turn(game)
        if result == "win":
            print("Congratulations! You won the battle!")
            break
        if result == "lose" or getattr(player, "health", 1) <= 0:
            print("You have been defeated. Game Over.")
            break

        enemy_turn(game, turn)

        if player is None:
            raise ValueError("Player is not set")

        if player.current_health <= 0:
            print("You have been defeated after the enemy's turn. Game Over.")
            break
        turn += 1
    print_separator()
    print("Game Over.")


if __name__ == "__main__":
    main()
