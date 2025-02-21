from collections.abc import Callable
from typing import Any

from SampleEfficientRL.Envs.Deckbuilder.EnvAction import EnvAction
from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import \
    DeckbuilderSingleBattleEnv

EffectCallback = Callable[[DeckbuilderSingleBattleEnv, EnvAction], EnvAction]
