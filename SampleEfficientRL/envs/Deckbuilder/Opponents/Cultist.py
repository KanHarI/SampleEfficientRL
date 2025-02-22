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
    def __init__(self, env: DeckbuilderSingleBattleEnv, max_health: int):
        # Uniform random max hp between 40 and 55
        super().__init__(env, OpponentTypeUIDs.CULTIST, max_health)

    def select_move(self, num_turn: int) -> NextMove:
        """
        Cultist does ritual on first turn, then attacks every turn after that.
        """
        if num_turn == 0:
            return NextMove(NextMoveType.RITUAL, 4)
        else:
            return NextMove(NextMoveType.ATTACK, 2)
