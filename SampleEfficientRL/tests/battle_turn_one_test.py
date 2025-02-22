from SampleEfficientRL.Envs.Deckbuilder.IroncladStarterVsCultist import (
    IroncladStarterVsCultist,
)


def battle_turn_one_test() -> None:
    env = IroncladStarterVsCultist()
    env.reset()
