FLOOR = 0
WALL = 1


class GameMap:
    def __init__(self, grid):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])

    def wrap_position(self, pos):
        row, col = pos
        row = row % self.height
        col = col % self.width
        return row, col

    def is_inside(self, pos):
        row, col = pos
        return 0 <= row < self.height and 0 <= col < self.width

    def get_cell(self, pos):
        row, col = self.wrap_position(pos)
        return self.grid[row][col]

    def is_wall(self, pos):
        row, col = self.wrap_position(pos)
        return self.grid[row][col] == WALL

    def is_walkable(self, pos):
        row, col = self.wrap_position(pos)
        return self.grid[row][col] == FLOOR

    def get_neighbors(self, pos):
        row, col = pos

        possible_moves = [
            (row - 1, col),  # up
            (row + 1, col),  # down
            (row, col - 1),  # left
            (row, col + 1)   # right
        ]

        neighbors = []

        for next_pos in possible_moves:
            next_pos = self.wrap_position(next_pos)

            if self.is_walkable(next_pos):
                neighbors.append(next_pos)

        return neighbors

    def print_map(self, human_pos=None, monster_positions=None, item_positions=None):
        # this is just for testing in terminal
        if monster_positions is None:
            monster_positions = []

        if item_positions is None:
            item_positions = []

        for i in range(self.height):
            line = ""

            for j in range(self.width):
                pos = (i, j)

                if human_pos == pos:
                    line += "H"
                elif pos in monster_positions:
                    line += "M"
                elif pos in item_positions:
                    line += "I"
                elif self.grid[i][j] == WALL:
                    line += "#"
                else:
                    line += "."

            print(line)