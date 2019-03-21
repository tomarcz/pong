from agents.epsilon_tracker import EpsilonTracker


class LinearEpsilonTracker(EpsilonTracker):
    def __init__(self, start, end, decay_to):
        super(LinearEpsilonTracker, self).__init__()
        self.start = start
        self.end = end
        self.decay_to = decay_to

    def epsilon_function(self, x):
        return max(self.end, self.start + x * (self.end - self.start) / self.decay_to)
