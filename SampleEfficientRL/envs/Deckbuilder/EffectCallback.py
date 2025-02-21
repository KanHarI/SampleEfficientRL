from collections.abc import Callable
from typing import Optional

from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import \
    DeckbuilderSingleBattleEnv

EffectCallback = Callable[[DeckbuilderSingleBattleEnv, Optional[int]], None]
