from typing import Dict, Tuple

from SampleEfficientRL.Envs.Deckbuilder.DeckbuilderSingleBattleEnv import (
    DeckbuilderSingleBattleEnv,
)
from SampleEfficientRL.Envs.Deckbuilder.Status import Status, StatusUIDs


class Entity:
    statuses: Dict[StatusUIDs, Tuple[Status, int]]

    def __init__(self, env: DeckbuilderSingleBattleEnv, max_health: int) -> None:
        self.env = env
        self.statuses = {}
        self.max_health = max_health
        self.current_health = max_health

    def get_active_statuses(self) -> Dict[StatusUIDs, Tuple[Status, int]]:
        return self.statuses

    def apply_status(self, status: Status, amount: int) -> None:
        if status.status_uid in self.statuses:
            status, old_amount = self.statuses[status.status_uid]
            new_amount = old_amount + amount
            if new_amount > 0:
                self.statuses[status.status_uid] = (status, new_amount)
            else:
                del self.statuses[status.status_uid]
        else:
            self.statuses[status.status_uid] = (status, amount)

    def reset_status(self, status_sid: StatusUIDs) -> None:
        if status_sid in self.statuses:
            del self.statuses[status_sid]

    def reduce_health(self, amount: int) -> bool:
        """
        Returns True if the entity is dead, False otherwise.
        """
        self.current_health -= amount
        if self.current_health <= 0:
            self.current_health = 0
            return True
        return False
