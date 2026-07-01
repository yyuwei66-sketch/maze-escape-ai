from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple
import random

Pos = Tuple[int, int]

_DIRS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

__all__ = ["TorusGrid", "GreedyMonsterAI", "make_greedy_controller",
           "make_greedy_controller_from_map", "pick_monster_spawns"]


class TorusGrid:

    def __init__(self, width: int, height: int, walls: Iterable[Pos] = ()):
        self.w = int(width)
        self.h = int(height)
        self.n = self.w * self.h
        self.set_walls(walls)

    def _i(self, x: int, y: int) -> int:
        return y * self.w + x

    def index(self, pos: Pos) -> int:
        x, y = pos
        return (y % self.h) * self.w + (x % self.w)

    def coord(self, i: int) -> Pos:
        return (i % self.w, i // self.w)

    def set_walls(self, walls: Iterable[Pos]) -> None:
        wallset = {(int(x) % self.w, int(y) % self.h) for (x, y) in walls}
        self.walls = wallset
        self.blocked = [False] * self.n
        for (x, y) in wallset:
            self.blocked[self._i(x, y)] = True
        self.adj: List[List[int]] = [[] for _ in range(self.n)]
        for y in range(self.h):
            for x in range(self.w):
                i = self._i(x, y)
                if self.blocked[i]:
                    continue
                for dx, dy in _DIRS:
                    j = ((y + dy) % self.h) * self.w + ((x + dx) % self.w)
                    if not self.blocked[j]:
                        self.adj[i].append(j)
        self.adj_stay: List[List[int]] = [
            (self.adj[i] + [i]) if not self.blocked[i] else []
            for i in range(self.n)
        ]
        self._dist_cache: Dict[int, List[int]] = {}

    def distance_field(self, src: Pos) -> List[int]:
        s = self.index(src)
        cached = self._dist_cache.get(s)
        if cached is not None:
            return cached
        dist = [-1] * self.n
        dist[s] = 0
        if not self.blocked[s]:
            q = deque((s,))
            adj = self.adj
            while q:
                c = q.popleft()
                d = dist[c] + 1
                for j in adj[c]:
                    if dist[j] < 0:
                        dist[j] = d
                        q.append(j)
        self._dist_cache[s] = dist
        return dist

    def toroidal_manhattan(self, a: Pos, b: Pos) -> int:
        dx = (a[0] - b[0]) % self.w
        dx = min(dx, self.w - dx)
        dy = (a[1] - b[1]) % self.h
        dy = min(dy, self.h - dy)
        return dx + dy


class GreedyMonsterAI:

    def __init__(self, grid: TorusGrid, steps_per_turn: int = 2,
                 allow_stay: bool = False, avoid_stacking: bool = True,
                 rng: random.Random | None = None):
        self.g = grid
        self.steps = steps_per_turn
        self.allow_stay = allow_stay
        self.avoid_stacking = avoid_stacking
        self.rng = rng or random.Random(0)
        self._prev: Dict[int, int] = {}          # last cell per monster (anti-jitter)

    def set_walls(self, walls: Iterable[Pos]) -> None:
        self.g.set_walls(walls)

    def decide(self, player_pos: Pos,
               monster_positions: Sequence[Pos]) -> List[List[Pos]]:
        g = self.g
        p_idx = g.index(player_pos)
        occupied = ({g.index(m) for m in monster_positions}
                    if self.avoid_stacking else set())
        paths: List[List[Pos]] = []
        for mi, m in enumerate(monster_positions):
            cur = g.index(m)
            occupied.discard(cur)
            path = [g.coord(cur)]
            for _ in range(self.steps):
                if cur == p_idx:
                    break
                nxt = self._best_step(cur, p_idx, occupied, mi)
                self._prev[mi] = cur             # remember where we came from
                cur = nxt
                path.append(g.coord(cur))
                if cur == p_idx:
                    break
            occupied.add(cur)
            paths.append(path)
        return paths

    def _best_step(self, cur: int, p_idx: int, occupied: set, mi: int) -> int:
        """Manhattan-only greedy step. BFS is NOT used for the decision."""
        g = self.g
        p = g.coord(p_idx)

        # legal neighbours (adj already excludes walls)
        cands = list(g.adj[cur])
        if self.allow_stay:
            cands.append(cur)
        if not cands:
            return cur                            # rule 3: walled in -> stay

        # rule 1: avoid stepping straight back to the previous cell
        prev = self._prev.get(mi)
        non_back = [j for j in cands if j != prev] or cands

        # core: rank by toroidal Manhattan distance only (+ anti-stacking)
        def dist(j: int) -> int:
            return g.toroidal_manhattan(g.coord(j), p)

        def key(j: int) -> Tuple[int, int]:
            penalty = 1 if (j in occupied and j != p_idx) else 0
            return (dist(j), penalty)

        best_key = min(key(j) for j in non_back)
        best_cands = [j for j in non_back if key(j) == best_key]

        if len(best_cands) == 1:
            return best_cands[0]
        # rule 2: on a tie, take the neighbour that most points at the player
        return self._prefer_toward(best_cands, cur, p_idx)

    def _prefer_toward(self, cands: List[int], cur: int, p_idx: int) -> int:
        g = self.g
        W, H = g.w, g.h
        cx, cy = g.coord(cur)
        px, py = g.coord(p_idx)

        def signed(delta: int, size: int) -> int:
            if delta > size // 2:
                delta -= size
            if delta < -size // 2:
                delta += size
            return delta

        tx, ty = signed(px - cx, W), signed(py - cy, H)

        def score(j: int) -> int:
            jx, jy = g.coord(j)
            mx, my = signed(jx - cx, W), signed(jy - cy, H)
            s = 0
            if mx * tx > 0:
                s += abs(tx)
            if my * ty > 0:
                s += abs(ty)
            return s

        return max(cands, key=score)


def make_greedy_controller(walls: Iterable[Pos] = (),
                           width: int = 30, height: int = 30,
                           steps_per_turn: int = 2,
                           allow_stay: bool = False,
                           avoid_stacking: bool = True) -> GreedyMonsterAI:
    grid = TorusGrid(width, height, walls)
    return GreedyMonsterAI(grid, steps_per_turn=steps_per_turn,
                           allow_stay=allow_stay, avoid_stacking=avoid_stacking)


import os


def _find_map_file(path=None):
    candidates = ([path] if path else []) + [
        "map/generated_map.txt", "../map/generated_map.txt",
        "data/generated_map.txt", "generated_map.txt",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _load_map_txt(path: str) -> Tuple[List[List[int]], List[Pos], List[Pos]]:
    raw_rows: List[List[int]] = []
    raw_spawns: List[Tuple[int, int]] = []

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    reading_grid = True
    width = None
    for lineno, line in enumerate(lines, start=1):
        parts = line.split()
        try:
            values = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"invalid integer on line {lineno}: {line!r}") from exc

        is_grid_row = (
            reading_grid
            and len(values) > 2
            and all(value in (0, 1) for value in values)
        )
        if is_grid_row:
            if width is None:
                width = len(values)
            elif len(values) != width:
                raise ValueError(
                    f"inconsistent map width on line {lineno}: "
                    f"expected {width}, got {len(values)}"
                )
            raw_rows.append(values)
            continue

        reading_grid = False
        if len(values) != 2:
            raise ValueError(
                f"spawn line {lineno} must contain exactly two integers"
            )
        raw_spawns.append((values[0], values[1]))

    if not raw_rows:
        raise ValueError("map file does not contain any grid rows")
    if not raw_spawns:
        raise ValueError("map file does not contain any spawn coordinates")

    height = len(raw_rows)
    width = len(raw_rows[0])
    walls = [
        (col, row)
        for row, cells in enumerate(raw_rows)
        for col, value in enumerate(cells)
        if value == 1
    ]
    spawns = [(col % width, row % height) for row, col in raw_spawns]
    return raw_rows, walls, spawns


def _write_map_txt(path: str, grid: List[List[int]], spawns: Sequence[Pos]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in grid:
            f.write(" ".join(str(cell) for cell in row) + " \n")
        for x, y in spawns:
            f.write(f"{y} {x}\n")


def pick_monster_spawns(grid: TorusGrid, player_pos: Pos, count: int,
                        min_dist: int = 8, exclude: Iterable[Pos] = (),
                        rng: random.Random | None = None) -> List[Pos]:
    if count <= 0:
        return []
    rng = rng or random.Random()
    field = grid.distance_field(player_pos)          
    taken = {player_pos, *exclude}
    free = [grid.coord(i) for i in range(grid.n)
            if not grid.blocked[i] and grid.coord(i) not in taken]
    rng.shuffle(free)
    chosen: List[Pos] = []
    for c in free:                                   
        if field[grid.index(c)] >= min_dist and all(
                grid.toroidal_manhattan(c, o) >= 3 for o in chosen):
            chosen.append(c)
            if len(chosen) == count:
                return chosen
    for c in free:                                  
        if c not in chosen and field[grid.index(c)] >= 0:
            chosen.append(c)
            if len(chosen) == count:
                return chosen
    for c in free:                                   
        if c not in chosen:
            chosen.append(c)
            if len(chosen) == count:
                break
    return chosen


def make_greedy_controller_from_map(map_path=None, steps_per_turn: int = 2,
                                    allow_stay: bool = False,
                                    avoid_stacking: bool = True):
    path = _find_map_file(map_path)
    if path is None:
        raise FileNotFoundError(
            "generated_map.txt not found -- run genetic_map.py first.")
    grid, walls, spawns = _load_map_txt(path)
    height, width = len(grid), len(grid[0])
    ctrl = make_greedy_controller(walls, width, height,
                                  steps_per_turn=steps_per_turn,
                                  allow_stay=allow_stay,
                                  avoid_stacking=avoid_stacking)
    return ctrl, spawns


def move_monsters_in_map(map_path=None, steps_per_turn: int = 2,
                         allow_stay: bool = False,
                         avoid_stacking: bool = True) -> List[Pos]:
    path = _find_map_file(map_path)
    if path is None:
        raise FileNotFoundError(
            "generated_map.txt not found -- run genetic_map.py first.")

    grid, walls, spawns = _load_map_txt(path)
    if len(spawns) < 2:
        raise ValueError("map file must contain human and monster coordinates")

    height, width = len(grid), len(grid[0])
    ctrl = make_greedy_controller(walls, width, height,
                                  steps_per_turn=steps_per_turn,
                                  allow_stay=allow_stay,
                                  avoid_stacking=avoid_stacking)
    player = spawns[0]
    monsters = spawns[1:]
    paths = ctrl.decide(player, monsters)
    moved_monsters = [monster_path[-1] for monster_path in paths]
    _write_map_txt(path, grid, [player, *moved_monsters])
    return moved_monsters


if __name__ == "__main__":
    moved = move_monsters_in_map()
    print(f"GREEDY moved monsters: {moved}")
