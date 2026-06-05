import random
import json
import os
from collections import deque

WIDTH = 30
HEIGHT = 30
FLOOR = 0
WALL = 1


def create_random_map(wall_rate=0.28):
    grid = []

    for i in range(HEIGHT):
        row = []
        for j in range(WIDTH):
            if random.random() < wall_rate:
                row.append(WALL)
            else:
                row.append(FLOOR)
        grid.append(row)

    return grid


def get_neighbors(pos, grid):
    x, y = pos
    h = len(grid)
    w = len(grid[0])

    result = []

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx = (x + dx) % h
        ny = (y + dy) % w

        if grid[nx][ny] == FLOOR:
            result.append((nx, ny))

    return result


def get_floor_cells(grid):
    cells = []

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == FLOOR:
                cells.append((i, j))

    return cells


def bfs_count(start, grid):
    q = deque([start])
    visited = {start}

    while q:
        cur = q.popleft()

        for nxt in get_neighbors(cur, grid):
            if nxt not in visited:
                visited.add(nxt)
                q.append(nxt)

    return len(visited)


def bfs_distance(start, end, grid):
    if start == end:
        return 0

    q = deque([(start, 0)])
    visited = {start}

    while q:
        cur, dist = q.popleft()

        for nxt in get_neighbors(cur, grid):
            if nxt == end:
                return dist + 1

            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, dist + 1))

    return None


def fitness(grid):
    floor_cells = get_floor_cells(grid)

    if len(floor_cells) < 2:
        return 0

    total_cells = WIDTH * HEIGHT
    wall_count = total_cells - len(floor_cells)
    wall_ratio = wall_count / total_cells

    # 1. connectivity score
    reachable = bfs_count(floor_cells[0], grid)
    reachable_ratio = reachable / len(floor_cells)
    reach_score = reachable_ratio * 40

    # 2. wall ratio score
    ideal_wall_ratio = 0.28
    wall_score = max(
        0,
        1 - abs(wall_ratio - ideal_wall_ratio) / ideal_wall_ratio
    ) * 20

    # 3. junction score
    junctions = 0
    for cell in floor_cells:
        if len(get_neighbors(cell, grid)) >= 3:
            junctions += 1

    junction_ratio = junctions / len(floor_cells)
    junction_score = min(junction_ratio * 100, 20)

    # 4. path distance score
    distances = []
    for _ in range(30):
        a = random.choice(floor_cells)
        b = random.choice(floor_cells)
        d = bfs_distance(a, b, grid)

        if d is not None:
            distances.append(d)

    if len(distances) == 0:
        path_score = 0
    else:
        avg_dist = sum(distances) / len(distances)
        path_score = max(0, 1 - abs(avg_dist - 20) / 20) * 20

    return reach_score + wall_score + junction_score + path_score


def create_population(size):
    population = []

    for _ in range(size):
        population.append(create_random_map())

    return population


def select_parent(scored_maps):
    candidates = random.sample(scored_maps, 3)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def crossover(parent_a, parent_b):
    child = []

    if random.random() < 0.5:
        cut = random.randint(1, HEIGHT - 2)
        child = parent_a[:cut] + parent_b[cut:]
    else:
        cut = random.randint(1, WIDTH - 2)

        for i in range(HEIGHT):
            row = parent_a[i][:cut] + parent_b[i][cut:]
            child.append(row)

    return child


def mutate(grid, mutation_rate=0.01):
    new_grid = []

    for i in range(HEIGHT):
        row = []
        for j in range(WIDTH):
            value = grid[i][j]

            if random.random() < mutation_rate:
                value = 1 - value

            row.append(value)

        new_grid.append(row)

    return new_grid


def generate_map_ga(pop_size=30, generations=20, mutation_rate=0.01, elite_num=5):
    population = create_population(pop_size)

    best_map = None
    best_score = -1

    for gen in range(generations):
        scored = []

        for grid in population:
            score = fitness(grid)
            scored.append((score, grid))

            if score > best_score:
                best_score = score
                best_map = [row[:] for row in grid]

        scored.sort(key=lambda x: x[0], reverse=True)

        print(
            "Generation",
            gen + 1,
            "Current Best:",
            round(scored[0][0], 2),
            "Global Best:",
            round(best_score, 2)
        )

        next_population = []

        for i in range(elite_num):
            next_population.append([row[:] for row in scored[i][1]])

        while len(next_population) < pop_size:
            p1 = select_parent(scored)
            p2 = select_parent(scored)

            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)

            next_population.append(child)

        population = next_population

    return best_map


def torus_manhattan_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])

    return min(dx, HEIGHT - dx) + min(dy, WIDTH - dy)


def get_valid_spawn_points(grid, min_dist=10, max_dist=25):
    floor_cells = get_floor_cells(grid)

    for _ in range(1000):
        human = random.choice(floor_cells)
        monster = random.choice(floor_cells)

        if human == monster:
            continue

        d = bfs_distance(human, monster, grid)

        if d is not None and min_dist <= d <= max_dist:
            return human, monster

    return random.choice(floor_cells), random.choice(floor_cells)


def get_approx_torus_spawn_points(grid, target_dist=20, attempts=5000):
    floor_cells = get_floor_cells(grid)

    if len(floor_cells) < 2:
        raise ValueError("Need at least two floor cells to generate spawn points.")

    best_pair = None
    best_delta = None

    for _ in range(attempts):
        human = random.choice(floor_cells)
        monster = random.choice(floor_cells)

        if human == monster:
            continue

        dist = torus_manhattan_distance(human, monster)
        delta = abs(dist - target_dist)

        if delta == 0:
            return human, monster

        if best_delta is None or delta < best_delta:
            best_pair = (human, monster)
            best_delta = delta

    if best_pair is not None:
        return best_pair

    human, monster = random.sample(floor_cells, 2)
    return human, monster


def save_map_to_txt(grid, file_path="data/generated_map.txt", spawn_points=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for row in grid:
            line = " ".join(str(cell) for cell in row)
            f.write(line + "\n")

        if spawn_points is not None:
            for x, y in spawn_points:
                f.write(f"{x} {y}\n")

    print(f"TXT map saved to: {file_path}")
    

def print_map(grid):
    for row in grid:
        line = ""

        for cell in row:
            if cell == FLOOR:
                line += "."
            else:
                line += "#"

        print(line)


if __name__ == "__main__":
    generated_map = generate_map_ga(
        pop_size=30,
        generations=20,
        mutation_rate=0.01,
        elite_num=5
    )

    print_map(generated_map)
    print("Fitness:", fitness(generated_map))

    human_pos, monster_pos = get_approx_torus_spawn_points(generated_map)

    print("Human spawn:", human_pos)
    print("Monster spawn:", monster_pos)
    print("Spawn distance:", torus_manhattan_distance(human_pos, monster_pos))

    save_map_to_txt(
        generated_map,
        "map/generated_map.txt",
        (human_pos, monster_pos)
    )

    print(f"{human_pos[0]} {human_pos[1]}")
    print(f"{monster_pos[0]} {monster_pos[1]}")
