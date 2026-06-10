"""Public AI package facade for the Maze Escape Flask game.

The Flask API exposes all coordinates as ``{"row": int, "col": int}``.
Some controller modules use ``(x, y)`` internally; callers should convert at the
application boundary instead of leaking that difference into JSON.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Literal, Sequence, Tuple

from .astar import astar, get_next_step_single, get_next_steps_two
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
CppAlgorithm = Literal["bfs", "sa", "genetic_map"]

WIDTH = 30
HEIGHT = 30

PACKAGE_DIR = Path(__file__).resolve().parent
BFS_ESCAPE_SOURCE = PACKAGE_DIR / "bfs_escape.cpp"
SA_SOURCE = PACKAGE_DIR / "SA.cpp"
GENETIC_MAP_SOURCE = PACKAGE_DIR / "genetic_map.cpp"


def _cpp_target_path(stem: str, platform_name: str | None = None) -> Path:
    platform_name = platform_name or sys.platform
    suffix = ".exe" if platform_name == "win32" else ""
    return PACKAGE_DIR / f"{stem}{suffix}"


BFS_ESCAPE_EXECUTABLE = _cpp_target_path("bfs_escape")
SA_EXECUTABLE = _cpp_target_path("SA")
GENETIC_MAP_EXECUTABLE = _cpp_target_path("genetic_map")

_CPP_TARGETS = {
    "bfs": (BFS_ESCAPE_SOURCE, BFS_ESCAPE_EXECUTABLE),
    "sa": (SA_SOURCE, SA_EXECUTABLE),
    "genetic_map": (GENETIC_MAP_SOURCE, GENETIC_MAP_EXECUTABLE),
}


class CppAlgorithmError(RuntimeError):
    """Raised when a bundled C++ algorithm cannot be compiled or executed."""


def _find_cpp_compiler() -> str | None:
    candidates = []
    configured_compiler = os.environ.get("CXX")
    if configured_compiler:
        candidates.append(configured_compiler)
    candidates.extend(("g++", "clang++", "cl"))

    for candidate in candidates:
        compiler = shutil.which(candidate)
        if compiler:
            return compiler
    return None


def _uses_msvc_flags(compiler: str) -> bool:
    compiler_name = compiler.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return compiler_name in {"cl", "cl.exe", "clang-cl", "clang-cl.exe"}


def _cpp_compile_command(
    compiler: str,
    source: Path,
    executable: Path,
) -> list[str]:
    if _uses_msvc_flags(compiler):
        return [
            compiler,
            "/nologo",
            "/O2",
            "/EHsc",
            "/std:c++17",
            str(source),
            f"/Fe:{executable}",
        ]
    return [
        compiler,
        "-O2",
        "-std=c++17",
        str(source),
        "-o",
        str(executable),
    ]


def ensure_cpp_executable(algorithm: CppAlgorithm) -> Path:
    """Return a compiled executable for a bundled C++ algorithm.

    The source programs read ``../map/generated_map.txt`` relative to their
    working directory, so callers may execute the returned binary from any
    temporary ``ai`` directory that has a sibling ``map`` directory.
    """

    if algorithm not in _CPP_TARGETS:
        raise ValueError(f"unknown C++ algorithm: {algorithm}")

    source, executable = _CPP_TARGETS[algorithm]
    if (
        executable.exists()
        and executable.stat().st_mtime >= source.stat().st_mtime
    ):
        return executable

    compiler = _find_cpp_compiler()
    if compiler is None:
        raise CppAlgorithmError(
            "a C++17 compiler is required; install g++, clang++, or MSVC cl, "
            "or set the CXX environment variable"
        )

    result = subprocess.run(
        _cpp_compile_command(compiler, source, executable),
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
    algorithm: Literal["bfs", "sa"],
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


def run_cpp_genetic_map() -> tuple[list[list[int]], Pos, Pos]:
    """Generate a map and spawn points with the bundled C++ GA program."""

    executable = ensure_cpp_executable("genetic_map")

    with tempfile.TemporaryDirectory(prefix="maze_escape_ga_") as tmp:
        root = Path(tmp)
        map_dir = root / "map"
        map_dir.mkdir()
        map_path = map_dir / "generated_map.txt"

        result = subprocess.run(
            [str(executable)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise CppAlgorithmError(
                f"genetic map generation failed: {detail or 'unknown error'}"
            )
        if not map_path.exists():
            raise CppAlgorithmError(
                "genetic map generator did not create map/generated_map.txt"
            )

        return _read_genetic_map(map_path)


def _read_genetic_map(path: Path) -> tuple[list[list[int]], Pos, Pos]:
    lines = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines or len(lines[0]) != 2:
        raise CppAlgorithmError("generated map is missing its dimensions")

    try:
        height, width = (int(value) for value in lines[0])
    except ValueError as exc:
        raise CppAlgorithmError("generated map has invalid dimensions") from exc

    if (height, width) != (HEIGHT, WIDTH):
        raise CppAlgorithmError(
            f"generated map must be {HEIGHT}x{WIDTH}, got {height}x{width}"
        )
    if len(lines) < height + 3:
        raise CppAlgorithmError("generated map is missing grid rows or spawns")

    try:
        grid = [[int(value) for value in row] for row in lines[1:height + 1]]
        human = tuple(int(value) for value in lines[height + 1])
        monster = tuple(int(value) for value in lines[height + 2])
    except ValueError as exc:
        raise CppAlgorithmError("generated map contains a non-integer value") from exc

    if any(len(row) != width for row in grid):
        raise CppAlgorithmError("generated map has an invalid row width")
    if any(cell not in (0, 1) for row in grid for cell in row):
        raise CppAlgorithmError("generated map cells must be 0 or 1")
    if len(human) != 2 or len(monster) != 2:
        raise CppAlgorithmError("generated map must contain two spawn coordinates")

    human_pos = (human[0] % height, human[1] % width)
    monster_pos = (monster[0] % height, monster[1] % width)
    if grid[human_pos[0]][human_pos[1]] != 0:
        raise CppAlgorithmError("generated human spawn is inside a wall")
    if grid[monster_pos[0]][monster_pos[1]] != 0:
        raise CppAlgorithmError("generated monster spawn is inside a wall")
    if human_pos == monster_pos:
        raise CppAlgorithmError("generated spawn coordinates overlap")

    return grid, human_pos, monster_pos


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
    "GENETIC_MAP_EXECUTABLE",
    "GENETIC_MAP_SOURCE",
    "GreedyMonsterAI",
    "HEIGHT",
    "MinimaxMonsterAI",
    "PACKAGE_DIR",
    "SA_EXECUTABLE",
    "SA_SOURCE",
    "WIDTH",
    "astar",
    "ensure_cpp_executable",
    "get_next_step_single",
    "get_next_steps_two",
    "make_greedy_controller",
    "make_minimax_controller",
    "many_row_col_to_xy",
    "many_xy_to_row_col",
    "pick_greedy_monster_spawns",
    "pick_minimax_monster_spawns",
    "row_col_to_xy",
    "run_cpp_genetic_map",
    "run_cpp_map_algorithm",
    "walls_from_grid",
    "xy_to_row_col",
]
