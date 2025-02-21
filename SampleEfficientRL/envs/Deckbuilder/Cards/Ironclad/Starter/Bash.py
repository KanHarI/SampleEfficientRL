from SampleEfficientRL.Envs.Deckbuilder.Card import Card, CardType, CardUIDs


class Bash(Card):
    def __init__(self):
        super().__init__(card_type=CardType.ATTACK, cost=2, card_uid=CardUIDs.BASH)
