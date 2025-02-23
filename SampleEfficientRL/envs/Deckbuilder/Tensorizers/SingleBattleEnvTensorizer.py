

from dataclasses import dataclass
from enum import Enum
from typing import List
from SampleEfficientRL.Envs.Deckbuilder.Card import CardUIDs


# Warning: changing this list will invalidate all the pre-trained models weights
SUPPORTED_CARDS_UIDs: List[CardUIDs] = [
    CardUIDs.BASH,
    CardUIDs.STRIKE,
    CardUIDs.DEFEND,
]

class ENTITY_TYPE(Enum):
    PLAYER = 0
    ENEMY_1 = 1
    ENEMY_2 = 2
    ENEMY_3 = 3
    ENEMY_4 = 4
    ENEMY_5 = 5

class TokenType(Enum):
    DRAW_DECK_CARD = 0
    DISCARD_DECK_CARD = 1
    EXHAUST_DECK_CARD = 2
    ENTITY_HP = 3
    ENTITY_MAX_HP = 4
    ENTITY_ENERGY = 5
    ENTITY_STATUS = 5
    
    

@dataclass
class SingleBattleEnvTensorizerConfig:
    token_type_dims: int

