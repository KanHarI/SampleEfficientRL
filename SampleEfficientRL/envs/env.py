from abc import ABC, abstractmethod


class Env(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def observe(self):
        pass
