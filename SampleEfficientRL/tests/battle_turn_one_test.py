from SampleEfficientRL.Envs.Deckbuilder.IroncladStarterVsCultist import (
    IroncladStarterVsCultist,
)


def test_battle_turn_one() -> None:
    env = IroncladStarterVsCultist()
    env.reset()
