from __future__ import annotations

from types import SimpleNamespace
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import ai


class CppPortabilityTest(unittest.TestCase):
    def test_windows_targets_use_exe_suffix(self):
        target = ai._cpp_target_path("genetic_map", platform_name="win32")

        self.assertEqual(target, ai.PACKAGE_DIR / "genetic_map.exe")

    def test_posix_targets_do_not_use_exe_suffix(self):
        target = ai._cpp_target_path("genetic_map", platform_name="linux")

        self.assertEqual(target, ai.PACKAGE_DIR / "genetic_map")

    def test_find_cpp_compiler_prefers_cxx_environment_variable(self):
        with (
            patch.dict("os.environ", {"CXX": r"C:\tools\clang++.exe"}, clear=True),
            patch("ai.shutil.which", side_effect=lambda name: name),
        ):
            compiler = ai._find_cpp_compiler()

        self.assertEqual(compiler, r"C:\tools\clang++.exe")

    def test_find_cpp_compiler_falls_back_to_msvc(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "ai.shutil.which",
                side_effect=lambda name: r"C:\VS\cl.exe" if name == "cl" else None,
            ),
        ):
            compiler = ai._find_cpp_compiler()

        self.assertEqual(compiler, r"C:\VS\cl.exe")

    def test_msvc_compile_command_uses_windows_flags(self):
        command = ai._cpp_compile_command(
            r"C:\VS\cl.exe",
            Path(r"C:\project\ai\SA.cpp"),
            Path(r"C:\project\ai\SA.exe"),
        )

        self.assertIn("/std:c++17", command)
        self.assertIn("/EHsc", command)
        self.assertIn(r"/Fe:C:\project\ai\SA.exe", command)

    def test_gnu_compile_command_uses_portable_flags(self):
        command = ai._cpp_compile_command(
            "g++",
            Path("/project/ai/SA.cpp"),
            Path("/project/ai/SA"),
        )

        self.assertEqual(
            command,
            [
                "g++",
                "-O2",
                "-std=c++17",
                "/project/ai/SA.cpp",
                "-o",
                "/project/ai/SA",
            ],
        )

    def test_write_cpp_map_omits_sa_metadata_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generated_map.txt"

            ai._write_cpp_map(path, [[0]], (0, 0), (0, 0))

            self.assertEqual(path.read_text(encoding="utf-8").splitlines(), [
                "0",
                "0 0",
                "0 0",
            ])

    def test_write_cpp_map_includes_sa_previous_move_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generated_map.txt"

            ai._write_cpp_map(
                path,
                [[0]],
                (0, 0),
                (0, 0),
                sa_previous_move=((1, 2), (3, 4)),
            )

            self.assertEqual(path.read_text(encoding="utf-8").splitlines()[-1], (
                "1 1 2 3 4"
            ))

    def test_write_cpp_map_includes_bfs_previous_human_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generated_map.txt"

            ai._write_cpp_map(
                path,
                [[0]],
                (0, 0),
                (0, 0),
                bfs_previous_human=(5, 6),
            )

            self.assertEqual(path.read_text(encoding="utf-8").splitlines()[-1], (
                "2 5 6"
            ))

    def test_run_cpp_map_algorithm_passes_metadata_to_matching_algorithm(self):
        previous_move = ((1, 2), (3, 4))
        previous_human = (5, 6)
        grid = [[0 for _ in range(ai.WIDTH)] for _ in range(ai.HEIGHT)]
        completed = SimpleNamespace(returncode=0, stderr="", stdout="")

        with (
            patch("ai.ensure_cpp_executable", return_value=Path("/tmp/SA")),
            patch("ai.subprocess.run", return_value=completed),
            patch("ai._read_cpp_map", return_value=(grid, [(0, 0), (0, 1)])),
            patch("ai._write_cpp_map") as write_map,
        ):
            ai.run_cpp_map_algorithm(
                "sa",
                grid,
                (0, 0),
                (0, 1),
                sa_previous_move=previous_move,
                bfs_previous_human=previous_human,
            )
            ai.run_cpp_map_algorithm(
                "bfs",
                grid,
                (0, 0),
                (0, 1),
                sa_previous_move=previous_move,
                bfs_previous_human=previous_human,
            )

        self.assertEqual(
            write_map.call_args_list[0].kwargs["sa_previous_move"],
            previous_move,
        )
        self.assertIsNone(write_map.call_args_list[0].kwargs["bfs_previous_human"])
        self.assertIsNone(write_map.call_args_list[1].kwargs["sa_previous_move"])
        self.assertEqual(
            write_map.call_args_list[1].kwargs["bfs_previous_human"],
            previous_human,
        )

    def test_sa_moves_when_a_path_exists_on_branchy_map(self):
        raw_grid = [
            "000100000000010011000000001000",
            "000001011000100000010000000000",
            "011000011111011010000100001000",
            "110010100110000000111100000010",
            "000001000011100010000100000010",
            "110000000000000100100110010010",
            "010001011001011000011000111000",
            "000001010000000000011010000101",
            "100000010011000100111000010000",
            "100001100001000000000000100100",
            "000001010010001001000110110010",
            "010101000001111000000001010000",
            "000000001001001100000001000001",
            "100010000100010011101001001000",
            "100000010100000001100001000000",
            "100100010110011100101011001100",
            "110000011110010011010000010000",
            "000000000000100010001001110000",
            "001001100100000011000000000000",
            "011000000001100010011000000100",
            "000011000001011100000010100000",
            "100101101000000010000000101001",
            "000000001001011001011100001000",
            "000110001000010101000011010001",
            "100100000010100000001000011000",
            "000000010000000011010000101001",
            "000000001010010001001011010000",
            "100100010001011111100000001000",
            "000000100100000010001000111100",
            "110000110110000100000100000000",
        ]
        grid = [[int(cell) for cell in row] for row in raw_grid]
        human = (23, 20)
        monster = (24, 27)

        path = ai.astar([list(row) for row in grid], monster, human)
        self.assertGreater(len(path), 1)

        for _ in range(20):
            _, moved_monster = ai.run_cpp_map_algorithm("sa", grid, human, monster)

            self.assertNotEqual(moved_monster, monster)
            self.assertEqual(grid[moved_monster[0]][moved_monster[1]], 0)


if __name__ == "__main__":
    unittest.main()
