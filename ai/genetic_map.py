from __future__ import annotations

from typing import Sequence

Pos = tuple[int, int]

_last_spawns: tuple[Pos, Pos] | None = None


def generate_map_ga(
    pop_size: int = 30,
    generations: int = 20,
    mutation_rate: float = 0.01,
    elite_num: int = 5,
) -> list[list[int]]:
    """Compatibility wrapper around the bundled C++ GA map generator.

    The C++ implementation owns its GA parameters, so the Python arguments are
    accepted for older callers but are not forwarded.
    """

    del pop_size, generations, mutation_rate, elite_num
    from . import run_cpp_genetic_map

    global _last_spawns
    grid, human, monster = run_cpp_genetic_map()
    _last_spawns = (human, monster)
    return grid


def get_approx_torus_spawn_points(grid: Sequence[Sequence[int]]) -> tuple[Pos, Pos]:
    """Return the spawns produced by the latest C++ GA call.

    Older evaluation/data-collection scripts ask for spawn points after calling
    ``generate_map_ga``. If called independently, choose two distant floor cells
    as a deterministic fallback.
    """

    if _last_spawns is not None:
        return _last_spawns

    floor = [
        (row, col)
        for row, cells in enumerate(grid)
        for col, value in enumerate(cells)
        if int(value) == 0
    ]
    if len(floor) < 2:
        raise ValueError("grid must contain at least two floor cells")

    human = floor[0]
    height = len(grid)
    width = len(grid[0])

    def torus_dist(pos: Pos) -> int:
        dr = abs(pos[0] - human[0])
        dc = abs(pos[1] - human[1])
        return min(dr, height - dr) + min(dc, width - dc)

    monster = max(floor[1:], key=torus_dist)
    return human, monster
