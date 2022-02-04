import matplotlib.cm as cm


class GridState:
    """
    Represents State in 2D GridWorld
    """

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.init_x = x
        self.init_y = y

    def reset(self):
        self.x = self.init_x
        self.y = self.init_y

    def set(self, x, y):
        self.x = x
        self.y = y

    def equals(self, other):
        return self.x == other.x and self.y == other.y


class GridStateRegion:
    """
    Represents rectangle set of States in 2D GridWorld.
    points = [[x1,y1], [x2, y2]],
    (x1, y1) = lower left point
    (x2, y2) = upper right point
    """

    def __init__(self, points):
        self.points = points
        if points is not None:
            self.start_point = (points[0][0] - 0.5, points[0][1] - 0.5)
            self.width = points[1][0] - points[0][0] + 1
            self.height = points[1][1] - points[0][1] + 1

    def contains(self, pos: GridState):
        if self.points is None:
            return False

        if self.points[0][0] <= pos.x <= self.points[1][0] and \
                self.points[0][1] <= pos.y <= self.points[1][1]:
            return True
        return False


class HeatmapRegion(GridStateRegion):
    """
    Represents rectangle set of States in 2D GridWorld with temperature level.
    points   = [[x1,y1], [x2, y2]]
    heat_lvl = number of random moves added for an agent started its movement in this HeatmapRegion.
    """
    heat_lvl_max = 4
    lvl_colors = ["#410996", "#e24848", "#c97e38", "#c9c738"]
    cmap_colors = [0.1, 0.5, 0.7, 0.9, 1.0]

    def __init__(self, points, heat_lvl):
        super().__init__(points)
        self.heat_lvl = heat_lvl
        if 0 <= heat_lvl <= self.heat_lvl_max:
            self.color = cm.get_cmap("plasma")(self.cmap_colors[heat_lvl])
        else:
            self.color = "#ffffff"

