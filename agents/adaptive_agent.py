import random
from DRLH.agents import Agent


class Adaptive_agent(Agent):
    def __init__(self, reaction_factor=0.3, segment_size=50, n_actions=29):
        super(Adaptive_agent, self).__init__(n_actions=n_actions)
        self.reaction_factor = reaction_factor
        self.segment_size = segment_size
        self.operators_counts = [0] * n_actions
        self.operators_scores = [0.0] * n_actions
        self.operators_probs = [1 / n_actions for _ in range(n_actions)]
        self.step = 0

    def choose_action(self):
        op_ind = random.choices(range(self.n_actions), weights=self.operators_probs)[0]
        return op_ind

    def update(self, action, reward):
        self.operators_counts[action] += 1
        self.operators_scores[action] += reward
        self.step += 1
        if self.step % self.segment_size == 0:
            for j in range(self.n_actions):
                if self.operators_counts[j] == 0:
                    continue
                self.operators_probs[j] = self.operators_probs[j] * (1 - self.reaction_factor) + self.reaction_factor * (
                            self.operators_scores[j] / self.operators_counts[j])
            self.operators_counts = [0] * self.n_actions
            self.operators_scores = [0.0] * self.n_actions
            sum_operator_probs = sum(self.operators_probs)
            self.operators_probs = [prob / sum_operator_probs for prob in self.operators_probs]

    def reset(self):
        self.operators_counts = [0] * self.n_actions
        self.operators_scores = [0.0] * self.n_actions
        self.operators_probs = [1 / self.n_actions for _ in range(self.n_actions)]
        self.step = 0



