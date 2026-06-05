"""
monster_ai.py
=============

Monster pursuit AI for a toroidal (wrap-around) grid-pursuit game.

This module is a *standalone* component meant to plug into a teammate's
Pygame project.  It implements two monster strategies for the course
"Principles of AI":

    * "greedy"  - each monster descends the true shortest-path distance
                  field toward the player (wall-aware, very fast).
    * "minimax" - adversarial search: monsters are the MIN player (want to
                  minimise distance / capture), the human is the MAX player
                  (wants to survive).  Depth-limited alpha-beta with move
                  ordering and a cached BFS distance field for evaluation.
                  Supports one or several cooperating monsters.

------------------------------------------------------------------------
INTEGRATION CONTRACT  (this is all the teammate's main loop needs)
------------------------------------------------------------------------
Coordinates are (x, y) tuples, x = column in [0, width), y = row in
[0, height).  The board wraps around on both axes (no boundaries).

    from monster_ai import make_monster_controller

    ctrl = make_monster_controller(
        algorithm="minimax",        # or "greedy"
        walls=set_of_blocked_cells, # iterable of (x, y) wall cells
        width=30, height=30,
        steps_per_turn=2,           # monster moves 2 cells per turn
        depth=2,                    # minimax look-ahead in rounds
    )

    # Each game turn, AFTER the player has moved one cell:
    paths = ctrl.decide(player_pos, monster_positions)
    #   player_pos        : (x, y)
    #   monster_positions : [(x, y), (x, y), ...]   (1 or more monsters)
    #   paths             : [[(x,y), (x,y), (x,y)], ...]
    #                       paths[i] is monster i's route this turn,
    #                       starting at its current cell, one entry per
    #                       step.  The new position is paths[i][-1].
    #                       (Use the intermediate cells to animate.)

The contract is identical for "greedy" and "minimax", so you can swap the
algorithm per level without touching the game loop.

If the genetic-algorithm map regenerates, just build a new controller
(or call ctrl.set_walls(new_walls)).
------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple
import random

Pos = Tuple[int, int]

# Movement directions on the grid (x, y).  4-connected: E, W, S, N.
_DIRS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

_INF = float("inf")
_CAUGHT = 1_000_000  # magnitude for a capture; dwarfs any distance score

__all__ = [
    "TorusGrid",
    "GreedyMonsterAI",
    "MinimaxMonsterAI",
    "make_monster_controller",
]


# ======================================================================
#  Grid: geometry, obstacles and shortest-path distances
# ======================================================================
class TorusGrid:
    """A fixed wrap-around grid with obstacles.

    The grid never changes during a game, so BFS distance fields are
    cached and reused across turns and across the whole minimax search.
    """

    def __init__(self, width: int, height: int, walls: Iterable[Pos] = ()):
        self.w = int(width)
        self.h = int(height)
        self.n = self.w * self.h
        self.set_walls(walls)

    # ---- index <-> (x, y) helpers -----------------------------------
    def _i(self, x: int, y: int) -> int:
        return y * self.w + x

    def index(self, pos: Pos) -> int:
        x, y = pos
        return (y % self.h) * self.w + (x % self.w)

    def coord(self, i: int) -> Pos:
        return (i % self.w, i // self.w)

    # ---- (re)build obstacle layout ----------------------------------
    def set_walls(self, walls: Iterable[Pos]) -> None:
        wallset = {(int(x) % self.w, int(y) % self.h) for (x, y) in walls}
        self.walls = wallset
        self.blocked = [False] * self.n
        for (x, y) in wallset:
            self.blocked[self._i(x, y)] = True
        # Precompute neighbour indices for every free cell (hot path).
        self.adj: List[List[int]] = [[] for _ in range(self.n)]
        for y in range(self.h):
            for x in range(self.w):
                i = self._i(x, y)
                if self.blocked[i]:
                    continue
                row = self.adj[i]
                for dx, dy in _DIRS:
                    j = ((y + dy) % self.h) * self.w + ((x + dx) % self.w)
                    if not self.blocked[j]:
                        row.append(j)
        # Same lists but including "stay in place" (no allocation in search).
        self.adj_stay: List[List[int]] = [
            (self.adj[i] + [i]) if not self.blocked[i] else []
            for i in range(self.n)
        ]
        self._dist_cache: Dict[int, List[int]] = {}

    # ---- distances ---------------------------------------------------
    def distance_field(self, src: Pos) -> List[int]:
        """BFS shortest-path step counts from `src` to every cell.

        Wall- and wrap-aware.  Returns a list indexed by cell index;
        -1 means unreachable.  Cached per source.
        """
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
        """Wrap-aware Manhattan distance (ignores walls).  Cheap heuristic."""
        dx = (a[0] - b[0]) % self.w
        dx = min(dx, self.w - dx)
        dy = (a[1] - b[1]) % self.h
        dy = min(dy, self.h - dy)
        return dx + dy


# ======================================================================
#  GREEDY
# ======================================================================
class GreedyMonsterAI:
    """Greedy pursuit using the true shortest-path distance field.

    One BFS from the player per turn gives every cell's distance to the
    player; each monster then walks `steps_per_turn` cells, each step to
    the neighbour with the smallest distance.  Because it uses real
    shortest paths (not Manhattan) it never gets stuck behind a wall on a
    connected map.  Cost: O(N) per turn.
    """

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
            if d < 0:  # unreachable region -> fall back to wrap-Manhattan
                d = 10 ** 6 + g.toroidal_manhattan(g.coord(j), g.coord(p_idx))
            penalty = 1 if (j in occupied and j != p_idx) else 0
            key = (d, penalty)
            if key < best_key:
                best_key, best = key, j
        return best


# ======================================================================
#  MINIMAX (alpha-beta)
# ======================================================================
class MinimaxMonsterAI:
    """Adversarial pursuit search.

    Monsters are the MINimising player (small distance / capture is good);
    the human is the MAXimising player (large distance / survival is good).

    A "round" is modelled as: each monster makes `steps_per_turn` single
    moves, then the human makes one move.  Multiple monsters are searched
    sequentially within the monster phase, so monster #2 already "sees"
    monster #1's move -> emergent pincer / coordination.

    Optimisations:
      * alpha-beta pruning
      * move ordering (monsters try closing moves first, player tries the
        safest moves first) to maximise cut-offs
      * a persistent BFS distance cache shared by the leaf evaluation
    """

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

    # ---- public API --------------------------------------------------
    def decide(self, player_pos: Pos,
               monster_positions: Sequence[Pos]) -> List[List[Pos]]:
        g = self.g
        p = g.index(player_pos)
        monsters = tuple(g.index(m) for m in monster_positions)
        k = len(monsters)

        # Ply schedule for one round, repeated for the look-ahead horizon:
        # every monster moves `steps` cells, then the human moves once.
        round_sched: List[object] = []
        for mi in range(k):
            round_sched += [mi] * self.steps
        round_sched.append("P")
        self._sched = round_sched * max(1, self.depth_rounds)
        self._tt = {}                     # transposition table for THIS decision
        self._pfield = g.distance_field(player_pos)  # ordering hint, cached

        self._search(p, monsters, 0, -_INF, _INF)

        # Reconstruct this turn's monster moves by walking the table along
        # the principal variation (the move stored at each visited node).
        return self._extract_paths(p, monsters)

    # ---- alpha-beta with transposition table -------------------------
    def _search(self, p: int, monsters: Tuple[int, ...], idx: int,
                alpha: float, beta: float) -> float:
        # Terminal: capture (sooner = better for monsters).
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
            if flag == 0:                      # EXACT
                return val
            if flag == 1 and val >= beta:      # LOWER bound
                return val
            if flag == 2 and val <= alpha:     # UPPER bound
                return val
            tt_move = mv                       # use stored move for ordering

        agent = sched[idx]
        a0, b0 = alpha, beta

        if agent == "P":                       # ---- MAX (human) ----
            moves = self._order_player(self.g.adj_stay[p] if self.player_can_stay
                                       else self.g.adj[p], monsters, tt_move)
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

        # ---- MIN (monster `agent`) ----
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

    # ---- ordering (cheap, drives the pruning) ------------------------
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
            reverse=True,                       # safest first
        )
        if tt_move is not None and tt_move in moves:
            ordered.remove(tt_move)
            ordered.insert(0, tt_move)
        return ordered

    def _evaluate(self, p: int, monsters: Tuple[int, ...]) -> float:
        """Leaf score: larger = safer for the human (MAX)."""
        field = self.g.distance_field(self.g.coord(p))
        far = self.g.w + self.g.h
        ds = [(field[m] if field[m] >= 0 else far) for m in monsters]
        # Weight the nearest threat heavily; the sum keeps every monster
        # closing in instead of both chasing the same flank.
        return min(ds) * 4 + sum(ds)

    # ---- move extraction (walk the PV via the transposition table) ---
    def _extract_paths(self, p: int, monsters: Tuple[int, ...]) -> List[List[Pos]]:
        g = self.g
        k = len(monsters)
        paths = [[g.coord(m)] for m in monsters]
        mons = list(monsters)
        plies_this_turn = k * self.steps
        for idx in range(plies_this_turn):
            if any(m == p for m in mons):       # already captured -> stop
                break
            agent = self._sched[idx]
            entry = self._tt.get((p, tuple(mons), idx))
            if entry is None:
                break                            # fell outside searched line
            mv = entry[2]
            mons[agent] = mv
            paths[agent].append(g.coord(mv))
        return paths


# ======================================================================
#  Factory
# ======================================================================
def make_monster_controller(algorithm: str, walls: Iterable[Pos] = (),
                            width: int = 30, height: int = 30,
                            steps_per_turn: int = 2, depth: int = 2,
                            **kwargs):
    """Build a monster controller exposing `.decide(player_pos, monsters)`.

    algorithm : "greedy" | "minimax"
    walls     : iterable of (x, y) blocked cells
    depth     : minimax look-ahead in *rounds* (ignored for greedy)
    kwargs    : allow_stay / avoid_stacking (greedy);
                player_can_stay / monster_can_stay (minimax)
    """
    grid = TorusGrid(width, height, walls)
    algo = algorithm.lower()
    if algo == "greedy":
        return GreedyMonsterAI(
            grid, steps_per_turn=steps_per_turn,
            allow_stay=kwargs.get("allow_stay", False),
            avoid_stacking=kwargs.get("avoid_stacking", True),
        )
    if algo == "minimax":
        return MinimaxMonsterAI(
            grid, steps_per_turn=steps_per_turn, depth_rounds=depth,
            player_can_stay=kwargs.get("player_can_stay", True),
            monster_can_stay=kwargs.get("monster_can_stay", False),
        )
    raise ValueError(f"unknown algorithm {algorithm!r} (use 'greedy' or 'minimax')")
