from typing import Dict, List, Tuple

from SampleEfficientRL.Envs.Deckbuilder.Card import Card
from SampleEfficientRL.Envs.Deckbuilder.Status import Status, StatusUIDs


class Player:
    statuses: Dict[StatusUIDs, Tuple[Status, int]]

    def __init__(self, starting_deck: List[Card], max_health: int):
        self.deck = starting_deck
        self.max_health = max_health
        self.current_health = max_health
        self.statuses = {}

    def apply_status(self, status: Status, amount: int) -> None:
        if status.status_uid in self.statuses:
            status, old_amount = self.statuses[status.status_uid]
            self.statuses[status.status_uid] = (status, old_amount + amount)
        else:
            self.statuses[status.status_uid] = (status, amount)

    def remove_status(self, status_uid: StatusUIDs) -> None:
        if status_uid in self.statuses:
            del self.statuses[status_uid]
