import random

import chess


def test_chess():
    # Initialize a new chess board
    board = chess.Board()
    assert board is not None

    # Get the current state (FEN representation)
    state = board.fen()
    assert state is not None

    # Get legal moves
    legal_moves = list(board.legal_moves)
    assert len(legal_moves) > 0
    print(f"Legal moves: {legal_moves}")

    # Make a random move
    move = random.choice(legal_moves)
    board.push(move)

    # Verify the board state after move
    assert board.is_valid()
    print(f"Board state: {board.fen()}")
    print(f"Legal moves: {list(board.legal_moves)}")
