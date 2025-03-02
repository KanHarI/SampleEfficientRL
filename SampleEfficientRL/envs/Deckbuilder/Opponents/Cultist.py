from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
    DeckbuilderSingleBattleEnv,
)
from SampleEfficientRL.Envs.Deckbuilder.Opponent import (
    NextMove,
    NextMoveType,
    Opponent,
    OpponentTypeUIDs,
)


class Cultist(Opponent):
    # Define constants for move amounts
    RITUAL_AMOUNT = 4
    ATTACK_AMOUNT = 2

    def __init__(self, env: DeckbuilderSingleBattleEnv, max_health: int):
        # Uniform random max hp between 40 and 55
        super().__init__(env, OpponentTypeUIDs.CULTIST, max_health)

    def select_move(self, num_turn: int) -> NextMove:
        """
        Cultist does ritual on first turn, then attacks every turn after that.
        """
        if num_turn == 0:
            return NextMove(NextMoveType.RITUAL, self.RITUAL_AMOUNT)
        else:
            return NextMove(NextMoveType.ATTACK, self.ATTACK_AMOUNT)
