from abc import abstractmethod


class EpsilonTracker:
    def __init__(self):
        self.x = 0

    @abstractmethod
    def epsilon_function(self, x):
        pass

    def get(self):
        value = self.epsilon_function(self.x)
        self.x += 1
        return value

    def get_value(self):
        return self.epsilon_function(self.x)
