from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from SampleEfficientRL.Envs.Deckbuilder.EffectCallback import EffectCallback


class CardType(Enum):
    POWER = 1
    SKILL = 2
    ATTACK = 3
    CURSE = 4
    STATUS = 5


class CardUIDs(Enum):
    # Ironclad
    # Starter Cards
    BASH = 1001
    DEFEND = 1002
    STRIKE = 1003

    # Common Cards
    ANGER = 1104
    ARMAMENTS = 1105
    BODY_SLAM = 1106
    CLASH = 1107
    CLEAVE = 1108
    CLOTHESLINE = 1109
    FLEX = 1110
    HAVOC = 1111
    HEADBUTT = 1112
    HEAVY_BLADE = 1113
    IRON_WAVE = 1114
    PERFECTED_STRIKE = 1115
    POMMEL_STRIKE = 1116
    SHRUG_IT_OFF = 1117
    SWORD_BOOMERANG = 1118
    THUNDERCLAP = 1119
    TRUE_GRIT = 1120
    TWIN_STRIKE = 1121
    WARCRY = 1122
    WILD_STRIKE = 1123

    # Uncommon Cards
    BATTLE_TRANCE = 1224
    BLOOD_FOR_BLOOD = 1225
    BLOODLETTING = 1226
    BURNING_PACT = 1227
    CARNAGE = 1228
    COMBUST = 1229
    DARK_EMBRACE = 1230
    DISARM = 1231
    DROPKICK = 1232
    DUAL_WIELD = 1233
    ENTRENCH = 1234
    EVOLVE = 1235
    FEEL_NO_PAIN = 1236
    FIRE_BREATHING = 1237
    FLAME_BARRIER = 1238
    GHOSTLY_ARMOR = 1239
    HEMOKINESIS = 1240
    INFERNAL_BLADE = 1241
    INFLAME = 1242
    INTIMIDATE = 1243
    METALLICIZE = 1244
    POWER_THROUGH = 1245
    PUMMEL = 1246
    RAGE = 1247
    RAMPAGE = 1248
    RECKLESS_CHARGE = 1249
    RUPTURE = 1250
    SEARING_BLOW = 1251
    SECOND_WIND = 1252
    SEEING_RED = 1253
    SENTINEL = 1254
    SEVER_SOUL = 1255
    SHOCKWAVE = 1256
    SPOT_WEAKNESS = 1257
    UPPERCUT = 1258
    WHIRLWIND = 1259

    # Rare Cards
    BARRICADE = 1360
    BERSERK = 1361
    BLUDGEON = 1362
    BRUTALITY = 1363
    CORRUPTION = 1364
    DEMON_FORM = 1365
    DOUBLE_TAP = 1366
    EXHUME = 1367
    FEED = 1368
    FIEND_FIRE = 1369
    IMMOLATE = 1370
    IMPERVIOUS = 1371
    JUGGERNAUT = 1372
    LIMIT_BREAK = 1373
    OFFERING = 1374
    REAPER = 1375


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


class Card(ABC):
    def __init__(self, card_type: CardType, cost: int, card_uid: CardUIDs):
        self.card_type = card_type
        self.cost = cost
        self.card_uid = card_uid

    @abstractmethod
    def get_effects(self) -> Dict[CardEffectTrigger, EffectCallback]:
        pass

    @abstractmethod
    def upgrade(self) -> "Card":
        pass

    @abstractmethod
    def get_targeting_info(self) -> CardTargetingInfo:
        pass
