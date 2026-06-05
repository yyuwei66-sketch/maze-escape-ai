"""
greedy_ai.py
============
Greedy monster pursuit AI for a toroidal (wrap-around) grid game.

Each monster descends the true shortest-path distance field toward the
player (wall-aware BFS).  Fast: O(N) per turn.

INTEGRATION
-----------
    from greedy_ai import make_greedy_controller

    ctrl = make_greedy_controller(
        walls=set_of_blocked_cells,
        width=30, height=30,
        steps_per_turn=2,
    )

    # Each game turn, AFTER the player has moved one cell:
    paths = ctrl.decide(player_pos, monster_positions)
    #   player_pos        : (x, y)
    #   monster_positions : [(x, y), ...]
    #   paths             : [[(x,y), ...], ...]
    #                       paths[i] is monster i's route this turn.
    #                       The new position is paths[i][-1].
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple
import random

Pos = Tuple[int, int]

_DIRS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

__all__ = ["TorusGrid", "GreedyMonsterAI", "make_greedy_controller",
           "make_greedy_controller_from_map", "pick_monster_spawns"]


# ======================================================================
#  Grid
# ======================================================================
class TorusGrid:
    """A fixed wrap-around grid with obstacles."""

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
        """BFS shortest-path distances from src to every cell. -1 = unreachable."""
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


# ======================================================================
#  Greedy AI
# ======================================================================
class GreedyMonsterAI:
    """Greedy pursuit: each monster steps toward the smallest BFS distance."""

    def __init__(self, grid: TorusGrid, steps_per_turn: int = 2,
                 allow_stay: bool = False, avoid_stacking: bool = True,
                 rng: random.Random | None = None):
        self.g = grid
        self.steps = steps_per_turn
        self.allow_stay = allow_stay
        self.avoid_stacking = avoid_stacking
        self.rng = rng or random.Random(0)

    def set_walls(self, walls: Iterable[Pos]) -> None:
        self.g.set_walls(walls)

    def decide(self, player_pos: Pos,
               monster_positions: Sequence[Pos]) -> List[List[Pos]]:
        g = self.g
        field = g.distance_field(player_pos)
        p_idx = g.index(player_pos)
        occupied = ({g.index(m) for m in monster_positions}
                    if self.avoid_stacking else set())
        paths: List[List[Pos]] = []
        for m in monster_positions:
            cur = g.index(m)
            occupied.discard(cur)
            path = [g.coord(cur)]
            for _ in range(self.steps):
                if cur == p_idx:
                    break
                cur = self._best_step(cur, p_idx, field, occupied)
                path.append(g.coord(cur))
                if cur == p_idx:
                    break
            occupied.add(cur)
            paths.append(path)
        return paths

    def _best_step(self, cur: int, p_idx: int,
                   field: List[int], occupied: set) -> int:
        g = self.g
        cands = list(g.adj[cur])
        if self.allow_stay:
            cands.append(cur)
        best = cur
        best_key = (10 ** 18, 1)
        for j in cands:
            d = field[j]
            if d < 0:
                d = 10 ** 6 + g.toroidal_manhattan(g.coord(j), g.coord(p_idx))
            penalty = 1 if (j in occupied and j != p_idx) else 0
            key = (d, penalty)
            if key < best_key:
                best_key, best = key, j
        return best


# ======================================================================
#  Factory
# ======================================================================
def make_greedy_controller(walls: Iterable[Pos] = (),
                           width: int = 30, height: int = 30,
                           steps_per_turn: int = 2,
                           allow_stay: bool = False,
                           avoid_stacking: bool = True) -> GreedyMonsterAI:
    grid = TorusGrid(width, height, walls)
    return GreedyMonsterAI(grid, steps_per_turn=steps_per_turn,
                           allow_stay=allow_stay, avoid_stacking=avoid_stacking)


# ======================================================================
#  Genetic-map integration
# ======================================================================
#
# The map comes from the team's genetic_map.py.  Its load_map_txt() already
# returns walls and spawns in THIS module's convention -- (x, y) = (col, row),
# 0 = floor, 1 = wall -- so they drop straight in.  Use the helpers below and
# you never have to touch coordinates by hand.

import os


def _find_map_file(path=None):
    """Locate the generated map file in the usual places."""
    candidates = ([path] if path else []) + [
        "map/generated_map.txt", "data/generated_map.txt", "generated_map.txt",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def pick_monster_spawns(grid: TorusGrid, player_pos: Pos, count: int,
                        min_dist: int = 8, exclude: Iterable[Pos] = (),
                        rng: random.Random | None = None) -> List[Pos]:
    """Pick `count` monster spawns that are reachable from the player, at
    least `min_dist` away, and spread out.  The genetic map ships only one
    monster spawn, so this tops up to however many monsters the game needs.
    Relaxes the constraints on cramped maps so it always returns when free
    cells remain."""
    if count <= 0:
        return []
    rng = rng or random.Random()
    field = grid.distance_field(player_pos)          # -1 == unreachable
    taken = {player_pos, *exclude}
    free = [grid.coord(i) for i in range(grid.n)
            if not grid.blocked[i] and grid.coord(i) not in taken]
    rng.shuffle(free)
    chosen: List[Pos] = []
    for c in free:                                   # reachable + far + spread
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
    for c in free:                                   # anything (disconnected map)
        if c not in chosen:
            chosen.append(c)
            if len(chosen) == count:
                break
    return chosen


def make_greedy_controller_from_map(map_path=None, steps_per_turn: int = 2,
                                    allow_stay: bool = False,
                                    avoid_stacking: bool = True):
    """Build a greedy controller straight from the genetic-algorithm map.

    Returns (controller, spawns) where spawns[0] is the human spawn and
    spawns[1:] are the monster spawn(s) from the file, all as (x, y).
    """
    import genetic_map
    path = _find_map_file(map_path)
    if path is None:
        raise FileNotFoundError(
            "generated_map.txt not found -- run genetic_map.py first.")
    grid, walls, spawns = genetic_map.load_map_txt(path)
    height, width = len(grid), len(grid[0])
    ctrl = make_greedy_controller(walls, width, height,
                                  steps_per_turn=steps_per_turn,
                                  allow_stay=allow_stay,
                                  avoid_stacking=avoid_stacking)
    return ctrl, spawns


# ----------------------------------------------------------------------
#  Self-test: run the greedy monsters on the real GA map vs a fleeing human
# ----------------------------------------------------------------------
if __name__ == "__main__":
    NUM_MONSTERS = 2
    ctrl, spawns = make_greedy_controller_from_map()
    g = ctrl.g
    player = spawns[0]
    fld = g.distance_field(player)
    monsters = [m for m in spawns[1:] if fld[g.index(m)] >= 0][:NUM_MONSTERS - 1]
    monsters += pick_monster_spawns(g, player, NUM_MONSTERS - len(monsters),
                                    exclude=monsters)

    print(f"GREEDY on GA map  {g.w}x{g.h}, walls={len(g.walls)}")
    print(f"  human spawn   : {player}")
    print(f"  monster spawns: {monsters}")

    for turn in range(1, 201):
        # simulate a human that flees toward the farthest spot from monsters
        mfields = [g.distance_field(m) for m in monsters]
        pi = g.index(player)
        best, best_score = pi, -1
        for j in list(g.adj[pi]) + [pi]:
            s = min((f[j] if f[j] >= 0 else 0) for f in mfields)
            if s > best_score:
                best_score, best = s, j
        player = g.coord(best)
        if player in monsters:
            print(f"  CAUGHT at turn {turn}"); break
        paths = ctrl.decide(player, monsters)
        monsters = [p[-1] for p in paths]
        if any(player in p for p in paths):
            print(f"  CAUGHT at turn {turn}"); break
    else:
        print("  human survived 200 turns (likely a disconnected spawn)")
