from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
        DeckbuilderSingleBattleEnv,
    )


class CardType(Enum):
    POWER = 1
    SKILL = 2
    ATTACK = 3
    CURSE = 4
    STATUS = 5


class CardUIDs(Enum):
    # Curses
    ASCENDERS_BANE = 1001

    # Status cards
    SLIMED = 2001

    # Ironclad
    # Starter Cards
    BASH = 5001
    DEFEND = 5002
    STRIKE = 5003

    # Common Cards
    ANGER = 5104
    ARMAMENTS = 5105
    BODY_SLAM = 5106
    CLASH = 5107
    CLEAVE = 5108
    CLOTHESLINE = 5109
    FLEX = 5110
    HAVOC = 5111
    HEADBUTT = 5112
    HEAVY_BLADE = 5113
    IRON_WAVE = 5114
    PERFECTED_STRIKE = 5115
    POMMEL_STRIKE = 5116
    SHRUG_IT_OFF = 5117
    SWORD_BOOMERANG = 5118
    THUNDERCLAP = 5119
    TRUE_GRIT = 5120
    TWIN_STRIKE = 5121
    WARCRY = 5122
    WILD_STRIKE = 5123

    # Uncommon Cards
    BATTLE_TRANCE = 5224
    BLOOD_FOR_BLOOD = 5225
    BLOODLETTING = 5226
    BURNING_PACT = 5227
    CARNAGE = 5228
    COMBUST = 5229
    DARK_EMBRACE = 5230
    DISARM = 5231
    DROPKICK = 5232
    DUAL_WIELD = 5233
    ENTRENCH = 5234
    EVOLVE = 5235
    FEEL_NO_PAIN = 5236
    FIRE_BREATHING = 5237
    FLAME_BARRIER = 5238
    GHOSTLY_ARMOR = 5239
    HEMOKINESIS = 5240
    INFERNAL_BLADE = 5241
    INFLAME = 5242
    INTIMIDATE = 5243
    METALLICIZE = 5244
    POWER_THROUGH = 5245
    PUMMEL = 5246
    RAGE = 5247
    RAMPAGE = 5248
    RECKLESS_CHARGE = 5249
    RUPTURE = 5250
    SEARING_BLOW = 5251
    SECOND_WIND = 5252
    SEEING_RED = 5253
    SENTINEL = 5254
    SEVER_SOUL = 5255
    SHOCKWAVE = 5256
    SPOT_WEAKNESS = 5257
    UPPERCUT = 5258
    WHIRLWIND = 5259

    # Rare Cards
    BARRICADE = 5360
    BERSERK = 5361
    BLUDGEON = 5362
    BRUTALITY = 5363
    CORRUPTION = 5364
    DEMON_FORM = 5365
    DOUBLE_TAP = 5366
    EXHUME = 5367
    FEED = 5368
    FIEND_FIRE = 5369
    IMMOLATE = 5370
    IMPERVIOUS = 5371
    JUGGERNAUT = 5372
    LIMIT_BREAK = 5373
    OFFERING = 5374
    REAPER = 5375


class TargetType(Enum):
    OPPONENT = 1
    CARD = 2
    DISCARDED_CARD = 3
    EXHAUSTED_CARD = 4
    DRAW_PILE_CARD = 5


class CardEffectTrigger(Enum):
    ON_PLAY = 1
    ON_END_OF_TURN = 2
    ON_END_OF_TURN_DISCARD = 3
    ON_MIDTURN_DISCARD = 4
    ON_EXHAUST = 5
    ON_DRAW = 6


@dataclass
class CardTargetingInfo:
    requires_target: bool
    targeting_type: Optional[TargetType]


# Parameters: env, target_idx (if applicable)
CardEffectCallback = Callable[["DeckbuilderSingleBattleEnv", Optional[int]], None]


class Card(ABC):
    def __init__(self, card_type: CardType, cost: int, card_uid: CardUIDs):
        self.card_type = card_type
        self.cost = cost
        self.card_uid = card_uid

    @abstractmethod
    def get_effects(self) -> Dict[CardEffectTrigger, CardEffectCallback]:
        pass

    @abstractmethod
    def get_targeting_info(self) -> CardTargetingInfo:
        pass
