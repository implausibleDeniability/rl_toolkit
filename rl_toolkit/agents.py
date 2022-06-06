import numpy as np
import torch

from rl_toolkit.models import DQN


class Agent:
    """Base class for agents"""

    backbone = None

    def __init__(self, observation_space, action_space):
        self.action_space = action_space

    def sample_action():
        raise NotImplementedError("Abstract class Agent is used")

    def get_best_action():
        raise NotImplementedError("Abstract class Agent is used")

    def eval(
        self
    ):
        self.backbone.eval()

    def train(
        self
    ):
        self.backbone.train()


class DQNAgent(Agent):
    def __init__(self, observation_space, action_space):
        self.epsilon = 0.08
        self.action_space = action_space
        self.backbone = DQN(observation_space.shape[0], action_space.n)

    def sample_action(self, observation):
        if np.random.rand() < self.epsilon:
            action = self.action_space.sample()
        else:
            action = self.get_best_action(observation)
        return action

    def get_best_action(self, observation):
        q_values = self.backbone(torch.tensor(observation))
        best_action = q_values.argmax(axis=-1).numpy()
        return best_action
