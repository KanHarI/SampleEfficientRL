
from SampleEfficientRL.Envs.Deckbuilder.Card import Card, CardType, CardUIDs


class Defend(Card):
    def __init__(self):
        super().__init__(
            card_type=CardType.SKILL,
            cost=1,
            card_uid=CardUIDs.DEFEND
        )
