import heapq
import os
from typing import Optional

GRID_SIZE = 30


def wrap(pos: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    return r % GRID_SIZE, c % GRID_SIZE


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return dr + dc


def neighbors(grid: list[list[int]], pos: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    result = []
    for dr, dc in directions:
        nr, nc = wrap((r + dr, c + dc))
        if grid[nr][nc] == 0:
            result.append((nr, nc))
    return result


def astar(
    grid: list[list[int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:

    if start == goal:
        return [start]

    open_heap: list[tuple[int, int, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0 + heuristic(start, goal), 0, start))

    came_from: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
    g_score: dict[tuple[int, int], int] = {start: 0}

    while open_heap:
        f, g, current = heapq.heappop(open_heap)

        if current == goal:
            path = []
            node: Optional[tuple[int, int]] = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        if g > g_score.get(current, float("inf")):
            continue

        for nb in neighbors(grid, current):
            tentative_g = g + 1
            if tentative_g < g_score.get(nb, float("inf")):
                g_score[nb] = tentative_g
                came_from[nb] = current
                f_new = tentative_g + heuristic(nb, goal)
                heapq.heappush(open_heap, (f_new, tentative_g, nb))

    return []


def get_next_step_single(
    grid: list[list[int]],
    monster_pos: tuple[int, int],
    player_pos: tuple[int, int],
) -> tuple[int, int]:

    path = astar(grid, monster_pos, player_pos)
    if len(path) >= 2:
        return path[1]
    return monster_pos


def _intercept_point(
    grid: list[list[int]],
    monster_pos: tuple[int, int],
    player_pos: tuple[int, int],
    offset_steps: int = 4,
) -> tuple[int, int]:

    mr, mc = monster_pos
    pr, pc = player_pos

    dr = pr - mr
    dc = pc - mc

    if abs(dr) > GRID_SIZE // 2:
        dr = dr - GRID_SIZE if dr > 0 else dr + GRID_SIZE
    if abs(dc) > GRID_SIZE // 2:
        dc = dc - GRID_SIZE if dc > 0 else dc + GRID_SIZE

    sign_r = (1 if dr > 0 else -1) if dr != 0 else 0
    sign_c = (1 if dc > 0 else -1) if dc != 0 else 0

    target = wrap((pr + sign_r * offset_steps, pc + sign_c * offset_steps))
    tr, tc = target

    if grid[tr][tc] == 0:
        return target
    return player_pos


def get_next_steps_two(
    grid: list[list[int]],
    m1_pos: tuple[int, int],
    m2_pos: tuple[int, int],
    player_pos: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:

    dist1 = heuristic(m1_pos, player_pos)
    dist2 = heuristic(m2_pos, player_pos)

    if dist1 <= dist2:
        chaser_pos, blocker_pos = m1_pos, m2_pos
        chaser_is_m1 = True
    else:
        chaser_pos, blocker_pos = m2_pos, m1_pos
        chaser_is_m1 = False

    chaser_target = player_pos
    chaser_path = astar(grid, chaser_pos, chaser_target)
    chaser_next = chaser_path[1] if len(chaser_path) >= 2 else chaser_pos

    intercept_target = _intercept_point(grid, chaser_pos, player_pos, offset_steps=4)
    blocker_path = astar(grid, blocker_pos, intercept_target)
    blocker_next = blocker_path[1] if len(blocker_path) >= 2 else blocker_pos

    if chaser_is_m1:
        return chaser_next, blocker_next
    else:
        return blocker_next, chaser_next


# ──────────────────────────────────────────────
# map_loading and test entry
# ──────────────────────────────────────────────

if __name__ == "__main__":
    MAP_FILE_PATH = os.path.join("...", "map", "generated_map.txt")

    if not os.path.exists(MAP_FILE_PATH):
        MAP_FILE_PATH = os.path.join("map", "generated_map.txt")

    grid = []
    player_pos = (0, 0)
    m1_pos = (0, 0)
    m2_pos = (0, 0)

    try:
        with open(MAP_FILE_PATH, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        for i in range(GRID_SIZE):
            row_cells = [int(x) for x in lines[i].split()]
            grid.append(row_cells)

        p_row, p_col = [int(x) for x in lines[-2].split()]
        player_pos = (p_row, p_col)

      
        m1_row, m1_col = [int(x) for x in lines[-1].split()]
        m1_pos = (m1_row, m1_col)

        m2_row, m2_col = [int(x) for x in lines[-1].split()]
        m2_pos = (m2_row, m2_col)

