import numpy as np
from heatedgridworld.State import GridState


class GeneralPolicy:
    """
    Abstract class for eps-greedy policy and table Q.
    pos = position, current state.

    get_argsmax method has to be reimplemented.
    """

    def __init__(self, eps=0, act_size=None):
        self.type_prob = [1 - eps, eps]
        self.act_size = act_size
        self.rng = np.random.default_rng()

    def get_action(self, Q_val, pos):
        act_type = self.rng.choice([0, 1], p=self.type_prob)
        if act_type == 0:
            return self.get_greedy_action(Q_val, pos)
        return self.rng.choice(self.act_size)

    def get_greedy_action(self, Q_val, pos):
        # in case of several max values np.argmax returns only the first max
        argsmax = self.get_argsmax(Q_val, pos)
        probabilities = [1 / len(argsmax)] * len(argsmax)
        action = self.rng.choice(argsmax, p=probabilities)
        return action

    def get_argsmax(self, Q_val, pos):
        pass


class GridPolicy(GeneralPolicy):
    """
    Epsilon-greedy policy for 2D GridWorld.
    for Q_value = Q_val[position.x, position.y, action_id]
    """

    def get_argsmax(self, Q_val, pos: GridState):
        return np.argwhere(Q_val[pos.x, pos.y] == np.max(Q_val[pos.x, pos.y])).flatten()