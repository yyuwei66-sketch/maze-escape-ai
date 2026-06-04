import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from ai.genetic_map import generate_map_ga, get_valid_spawn_points, bfs_distance

grid = generate_map_ga(pop_size=10, generations=5)
print("Map size:", len(grid), "x", len(grid[0]))

human, monster = get_valid_spawn_points(grid)
print("Human:", human)
print("Monster:", monster)
print("Distance:", bfs_distance(human, monster, grid))