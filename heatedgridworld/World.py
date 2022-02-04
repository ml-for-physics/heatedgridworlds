import numpy as np
from heatedgridworld.State import GridState


class GridWorld:
    def __init__(self, size_x=1, size_y=1, bound='reflect', obstacles=None):
        self.size_x = size_x
        self.size_y = size_y
        self.bound = bound
        self.tmp_pos = GridState(x=None, y=None)
        if obstacles is None:
            obstacles = []
        self.obstacles = obstacles

        if size_x > 1 and size_y > 1:
            self.act_size = 4
        else:
            self.act_size = 2

    def __do_action(self, pos: GridState, new_pos: GridState, act: int = None):
        """internal: 0 left | 1 right | 2 down| 3 up |  """
        new_pos.set(pos.x, pos.y)
        shift = np.zeros(2)

        if self.bound == 'reflect':
            if act == 0:  # left
                new_pos.x = pos.x - 1 if pos.x > 0 else 0
            elif act == 1:  # right
                new_pos.x = pos.x + 1 if pos.x < self.size_x - 1 else self.size_x - 1
            elif act == 2:  # down
                new_pos.y = pos.y - 1 if pos.y > 0 else 0
            elif act == 3:  # up
                new_pos.y = pos.y + 1 if pos.y < self.size_y - 1 else self.size_y - 1

        elif self.bound == 'periodical':
            if act == 0:  # left
                if pos.x == 0:
                    new_pos.x = self.size_x - 1
                    shift[0] = -1
                else:
                    new_pos.x = pos.x - 1
            elif act == 1:  # right
                if pos.x == self.size_x - 1:
                    new_pos.x = 0
                    shift[0] = +1
                else:
                    new_pos.x = pos.x + 1
            elif act == 2:  # down
                if pos.y == 0:
                    new_pos.y = self.size_y - 1
                    shift[1] = -1
                else:
                    new_pos.y = pos.y - 1
            elif act == 3:  # up
                if pos.y == self.size_y - 1:
                    new_pos.y = 0
                    shift[1] = +1
                else:
                    new_pos.y = pos.y + 1

        for obstacle in self.obstacles:
            if obstacle.contains(new_pos):
                new_pos.set(pos.x, pos.y)
                break

        return shift

    def __do_actions(self, pos: GridState, new_pos: GridState, acts):
        """only: 0 left | 1 right | 2 down | 3 up """
        self.tmp_pos.set(pos.x, pos.y)
        shift = np.zeros(2)

        for act in acts:
            shift += self.__do_action(self.tmp_pos, new_pos, act)
            self.tmp_pos.set(new_pos.x, new_pos.y)

        return shift

    def do_action(self, pos: GridState, new_pos: GridState, act: int = None):
        """ 0 left | 1 right | 2 down | 3 up | 4 left, down | 5 right, down | 6 left, up | 7 right, up"""
        if (act is None
                or act == 0  # left
                or act == 1  # right
                or act == 2  # down
                or act == 3):  # up
            return self.__do_action(pos, new_pos, act)

        elif act == 4:  # left, down
            return self.__do_actions(pos, new_pos, [0, 2])
        elif act == 5:  # right, down
            return self.__do_actions(pos, new_pos, [1, 2])
        elif act == 6:  # left, up
            return self.__do_actions(pos, new_pos, [0, 3])
        elif act == 7:  # right, up
            return self.__do_actions(pos, new_pos, [1, 3])

    def get_name(self):
        return "%ix%i %s grid, %s actions" % (self.size_x, self.size_y, self.bound, self.act_size)

    def get_states_number(self):
        return self.size_x * self.size_y


class HeatmapGridWorld(GridWorld):
    """
    2D GridWorld with regions with different heat level T. Heat level adds T random movements
    after an agent's action. T is taken for a HeatmapRegion from which the agent starts its motion.
    """

    def __init__(self, size_x, size_y, bound='reflect', heat_regions=None, heat_off=False,
                 targets=None, step_reward=-1, target_reward=100, obstacles=None):
        super().__init__(size_x, size_y, bound, obstacles)
        if heat_regions is None or heat_off:
            heat_regions = []
        self.heat_regions = sorted(heat_regions, key=lambda region: -region.heat_lvl)
        if targets is None:
            targets = []
        self.targets = targets
        self.step_reward = step_reward
        self.target_reward = target_reward
        self.rng = np.random.default_rng()
        self.tmp_pos_1 = GridState(None, None)

    def do_action(self, pos: GridState, new_pos: GridState, act: int = None):
        target_reached = False
        super().do_action(pos, new_pos, act)

        for target_pos in self.targets:
            if target_pos.equals(new_pos):
                target_reached = True
                break
        # if HeatmapRegions overlap each other, heat_lvl is taken for the first
        # hottest region in the list (regions are sorted by T in descending order)
        if not target_reached:
            for region in self.heat_regions:
                if region.contains(pos):
                    for act in self.rng.choice(self.act_size, size=region.heat_lvl):
                        self.tmp_pos_1.set(new_pos.x, new_pos.y)
                        super().do_action(self.tmp_pos_1, new_pos, act)

                        for target_pos in self.targets:
                            if target_pos.equals(new_pos):
                                target_reached = True
                                break
                        if target_reached:
                            break
                    break
        reward = self.step_reward
        if target_reached:
            reward = self.target_reward
        return target_reached, reward

