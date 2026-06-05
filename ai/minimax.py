from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple
import random

Pos = Tuple[int, int]

_DIRS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

_INF = float("inf")
_CAUGHT = 1_000_000

__all__ = ["TorusGrid", "MinimaxMonsterAI", "make_minimax_controller",
           "make_minimax_controller_from_map", "pick_monster_spawns"]


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


class MinimaxMonsterAI:

    def __init__(self, grid: TorusGrid, steps_per_turn: int = 2,
                 depth_rounds: int = 2, player_can_stay: bool = True,
                 monster_can_stay: bool = False,
                 rng: random.Random | None = None):
        self.g = grid
        self.steps = steps_per_turn
        self.depth_rounds = depth_rounds
        self.player_can_stay = player_can_stay
        self.monster_can_stay = monster_can_stay
        self.rng = rng or random.Random(0)

    def set_walls(self, walls: Iterable[Pos]) -> None:
        self.g.set_walls(walls)

    def decide(self, player_pos: Pos,
               monster_positions: Sequence[Pos]) -> List[List[Pos]]:
        g = self.g
        p = g.index(player_pos)
        monsters = tuple(g.index(m) for m in monster_positions)
        k = len(monsters)

        round_sched: List[object] = []
        for mi in range(k):
            round_sched += [mi] * self.steps
        round_sched.append("P")
        self._sched = round_sched * max(1, self.depth_rounds)
        self._tt = {}
        self._pfield = g.distance_field(player_pos)

        self._search(p, monsters, 0, -_INF, _INF)
        return self._extract_paths(p, monsters)

    def _search(self, p: int, monsters: Tuple[int, ...], idx: int,
                alpha: float, beta: float) -> float:
        for m in monsters:
            if m == p:
                return -(_CAUGHT - idx)
        sched = self._sched
        if idx >= len(sched):
            return self._evaluate(p, monsters)

        key = (p, monsters, idx)
        tt = self._tt
        hit = tt.get(key)
        tt_move = None
        if hit is not None:
            flag, val, mv = hit
            if flag == 0:
                return val
            if flag == 1 and val >= beta:
                return val
            if flag == 2 and val <= alpha:
                return val
            tt_move = mv

        agent = sched[idx]
        a0, b0 = alpha, beta

        if agent == "P":
            moves = self._order_player(
                self.g.adj_stay[p] if self.player_can_stay else self.g.adj[p],
                monsters, tt_move)
            best_val, best_mv = -_INF, moves[0]
            for nm in moves:
                v = self._search(nm, monsters, idx + 1, alpha, beta)
                if v > best_val:
                    best_val, best_mv = v, nm
                if best_val > alpha:
                    alpha = best_val
                if alpha >= beta:
                    break
            flag = 1 if best_val >= b0 else (0 if best_val > a0 else 2)
            tt[key] = (flag, best_val, best_mv)
            return best_val

        mi = agent
        cur = monsters[mi]
        src = self.g.adj_stay[cur] if self.monster_can_stay else self.g.adj[cur]
        moves = self._order_monster(src, tt_move)
        best_val, best_mv = _INF, moves[0]
        for nm in moves:
            new_monsters = monsters[:mi] + (nm,) + monsters[mi + 1:]
            v = self._search(p, new_monsters, idx + 1, alpha, beta)
            if v < best_val:
                best_val, best_mv = v, nm
            if best_val < beta:
                beta = best_val
            if alpha >= beta:
                break
        flag = 2 if best_val <= a0 else (0 if best_val < b0 else 1)
        tt[key] = (flag, best_val, best_mv)
        return best_val

    def _order_monster(self, moves: List[int], tt_move) -> List[int]:
        f = self._pfield
        big = 10 ** 9
        ordered = sorted(moves, key=lambda j: f[j] if f[j] >= 0 else big)
        if tt_move is not None and tt_move in moves:
            ordered.remove(tt_move)
            ordered.insert(0, tt_move)
        return ordered

    def _order_player(self, moves: List[int], monsters: Tuple[int, ...],
                      tt_move) -> List[int]:
        g = self.g
        mons = [g.coord(m) for m in monsters]
        ordered = sorted(
            moves,
            key=lambda j: min(g.toroidal_manhattan(g.coord(j), m) for m in mons),
            reverse=True,
        )
        if tt_move is not None and tt_move in moves:
            ordered.remove(tt_move)
            ordered.insert(0, tt_move)
        return ordered

    def _evaluate(self, p: int, monsters: Tuple[int, ...]) -> float:
        field = self.g.distance_field(self.g.coord(p))
        far = self.g.w + self.g.h
        ds = [(field[m] if field[m] >= 0 else far) for m in monsters]
        return min(ds) * 4 + sum(ds)

    def _extract_paths(self, p: int, monsters: Tuple[int, ...]) -> List[List[Pos]]:
        g = self.g
        k = len(monsters)
        paths = [[g.coord(m)] for m in monsters]
        mons = list(monsters)
        plies_this_turn = k * self.steps
        for idx in range(plies_this_turn):
            if any(m == p for m in mons):
                break
            agent = self._sched[idx]
            entry = self._tt.get((p, tuple(mons), idx))
            if entry is None:
                break
            mv = entry[2]
            mons[agent] = mv
            paths[agent].append(g.coord(mv))
        return paths


def make_minimax_controller(walls: Iterable[Pos] = (),
                            width: int = 30, height: int = 30,
                            steps_per_turn: int = 2, depth: int = 2,
                            player_can_stay: bool = True,
                            monster_can_stay: bool = False) -> MinimaxMonsterAI:
    grid = TorusGrid(width, height, walls)
    return MinimaxMonsterAI(grid, steps_per_turn=steps_per_turn,
                            depth_rounds=depth,
                            player_can_stay=player_can_stay,
                            monster_can_stay=monster_can_stay)



import os


def _find_map_file(path=None):
    """Locate the generated map file in the usual places."""
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
    for c in free:                                   # just reachable
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


def make_minimax_controller_from_map(map_path=None, steps_per_turn: int = 2,
                                     depth: int = 2,
                                     player_can_stay: bool = True,
                                     monster_can_stay: bool = False):
    path = _find_map_file(map_path)
    if path is None:
        raise FileNotFoundError(
            "generated_map.txt not found -- run genetic_map.py first.")
    grid, walls, spawns = _load_map_txt(path)
    height, width = len(grid), len(grid[0])
    ctrl = make_minimax_controller(walls, width, height,
                                   steps_per_turn=steps_per_turn, depth=depth,
                                   player_can_stay=player_can_stay,
                                   monster_can_stay=monster_can_stay)
    return ctrl, spawns


def move_monsters_in_map(map_path=None, steps_per_turn: int = 2,
                         depth: int = 2,
                         player_can_stay: bool = True,
                         monster_can_stay: bool = False) -> List[Pos]:
    path = _find_map_file(map_path)
    if path is None:
        raise FileNotFoundError(
            "generated_map.txt not found -- run genetic_map.py first.")

    grid, walls, spawns = _load_map_txt(path)
    if len(spawns) < 2:
        raise ValueError("map file must contain human and monster coordinates")

    height, width = len(grid), len(grid[0])
    ctrl = make_minimax_controller(walls, width, height,
                                   steps_per_turn=steps_per_turn, depth=depth,
                                   player_can_stay=player_can_stay,
                                   monster_can_stay=monster_can_stay)
    player = spawns[0]
    monsters = spawns[1:]
    paths = ctrl.decide(player, monsters)
    moved_monsters = [monster_path[-1] for monster_path in paths]
    _write_map_txt(path, grid, [player, *moved_monsters])
    return moved_monsters


if __name__ == "__main__":
    moved = move_monsters_in_map()
    print(f"MINIMAX moved monsters: {moved}")
