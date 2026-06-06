"""Public AI package facade for the Maze Escape Flask game.

The Flask API exposes all coordinates as ``{"row": int, "col": int}``.
Some controller modules use ``(x, y)`` internally; callers should convert at the
application boundary instead of leaking that difference into JSON.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, Literal, Sequence, Tuple

from .astar import astar, get_next_step_single, get_next_steps_two
from .genetic_map import (
    HEIGHT,
    WIDTH,
    bfs_distance,
    fitness,
    generate_map_ga,
    get_approx_torus_spawn_points,
    get_valid_spawn_points,
)
from .greedy import (
    GreedyMonsterAI,
    make_greedy_controller,
    pick_monster_spawns as pick_greedy_monster_spawns,
)
from .minimax import (
    MinimaxMonsterAI,
    make_minimax_controller,
    pick_monster_spawns as pick_minimax_monster_spawns,
)

Pos = Tuple[int, int]
CppAlgorithm = Literal["bfs", "sa"]

PACKAGE_DIR = Path(__file__).resolve().parent
BFS_ESCAPE_SOURCE = PACKAGE_DIR / "bfs_escape.cpp"
SA_SOURCE = PACKAGE_DIR / "SA.cpp"
BFS_ESCAPE_EXECUTABLE = PACKAGE_DIR / "bfs_escape"
SA_EXECUTABLE = PACKAGE_DIR / "SA"

_CPP_TARGETS = {
    "bfs": (BFS_ESCAPE_SOURCE, BFS_ESCAPE_EXECUTABLE),
    "sa": (SA_SOURCE, SA_EXECUTABLE),
}


class CppAlgorithmError(RuntimeError):
    """Raised when a bundled C++ algorithm cannot be compiled or executed."""


def ensure_cpp_executable(algorithm: CppAlgorithm) -> Path:
    """Return a compiled executable for a bundled C++ algorithm.

    The source programs read ``../map/generated_map.txt`` relative to their
    working directory, so callers may execute the returned binary from any
    temporary ``ai`` directory that has a sibling ``map`` directory.
    """

    if algorithm not in _CPP_TARGETS:
        raise ValueError(f"unknown C++ algorithm: {algorithm}")

    source, executable = _CPP_TARGETS[algorithm]
    if executable.exists():
        return executable

    compiler = shutil.which("g++")
    if compiler is None:
        raise CppAlgorithmError("g++ is required to compile the C++ AI modules")

    result = subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(executable)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CppAlgorithmError(
            f"failed to compile {source.name}: {detail or 'unknown error'}"
        )
    return executable


def run_cpp_map_algorithm(
    algorithm: CppAlgorithm,
    grid: Sequence[Sequence[int]],
    human: Pos,
    monster: Pos,
) -> tuple[Pos, Pos]:
    """Run a C++ map-mutating algorithm and return ``(human, monster)``.

    The repository map file is not touched. A temporary ``map/generated_map.txt``
    is created, the executable is run from a temporary ``ai`` directory, and the
    resulting coordinates are read back.
    """

    executable = ensure_cpp_executable(algorithm)

    with tempfile.TemporaryDirectory(prefix="maze_escape_ai_") as tmp:
        root = Path(tmp)
        map_dir = root / "map"
        work_dir = root / "ai"
        map_dir.mkdir()
        work_dir.mkdir()
        map_path = map_dir / "generated_map.txt"
        _write_cpp_map(map_path, grid, human, monster)

        result = subprocess.run(
            [str(executable)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise CppAlgorithmError(
                f"{algorithm} failed: {detail or 'unknown error'}"
            )

        _, spawns = _read_cpp_map(map_path)
        if len(spawns) < 2:
            raise CppAlgorithmError(f"{algorithm} did not write two coordinates")
        return spawns[0], spawns[1]


def _write_cpp_map(
    path: Path,
    grid: Sequence[Sequence[int]],
    human: Pos,
    monster: Pos,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in grid:
            f.write(" ".join(str(int(cell)) for cell in row) + "\n")
        f.write(f"{human[0]} {human[1]}\n")
        f.write(f"{monster[0]} {monster[1]}\n")


def _read_cpp_map(path: Path) -> tuple[list[list[int]], list[Pos]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    grid = [[int(part) for part in line.split()] for line in lines[:HEIGHT]]
    spawns = [
        (int(parts[0]) % HEIGHT, int(parts[1]) % WIDTH)
        for parts in (line.split() for line in lines[HEIGHT:])
        if len(parts) == 2
    ]
    return grid, spawns


def walls_from_grid(grid: Sequence[Sequence[int]]) -> list[Pos]:
    """Return wall positions as ``(x, y)`` tuples for greedy/minimax modules."""

    return [
        (col, row)
        for row, cells in enumerate(grid)
        for col, value in enumerate(cells)
        if int(value) == 1
    ]


def row_col_to_xy(pos: Pos) -> Pos:
    row, col = pos
    return col, row


def xy_to_row_col(pos: Pos) -> Pos:
    x, y = pos
    return y, x


def many_row_col_to_xy(positions: Iterable[Pos]) -> list[Pos]:
    return [row_col_to_xy(pos) for pos in positions]


def many_xy_to_row_col(positions: Iterable[Pos]) -> list[Pos]:
    return [xy_to_row_col(pos) for pos in positions]


__all__ = [
    "BFS_ESCAPE_EXECUTABLE",
    "BFS_ESCAPE_SOURCE",
    "CppAlgorithmError",
    "GreedyMonsterAI",
    "HEIGHT",
    "MinimaxMonsterAI",
    "PACKAGE_DIR",
    "SA_EXECUTABLE",
    "SA_SOURCE",
    "WIDTH",
    "astar",
    "bfs_distance",
    "ensure_cpp_executable",
    "fitness",
    "generate_map_ga",
    "get_approx_torus_spawn_points",
    "get_next_step_single",
    "get_next_steps_two",
    "get_valid_spawn_points",
    "make_greedy_controller",
    "make_minimax_controller",
    "many_row_col_to_xy",
    "many_xy_to_row_col",
    "pick_greedy_monster_spawns",
    "pick_minimax_monster_spawns",
    "row_col_to_xy",
    "run_cpp_map_algorithm",
    "walls_from_grid",
    "xy_to_row_col",
]
