import random

class Agent:

    def __init__(self, n_actions):
        self.n_actions = n_actions

    def choose_action(self, observation):
        raise NotImplementedError

    def update(self, action, reward):
        pass

    def reset(self, info):
        pass

    def save_model(self, **kwargs):
        pass

    def learn(self):
        pass

    def load_model(self, **kwargs):
        pass

    def seed(self, seed=None):
        random.seed(seed)

    def train(self, **kwargs):
        pass

    def solve(self, problem):
        raise NotImplementedError