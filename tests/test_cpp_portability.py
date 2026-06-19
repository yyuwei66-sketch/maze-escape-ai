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

    def test_run_cpp_map_algorithm_passes_sa_metadata_only_to_sa(self):
        previous_move = ((1, 2), (3, 4))
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
            )
            ai.run_cpp_map_algorithm(
                "bfs",
                grid,
                (0, 0),
                (0, 1),
                sa_previous_move=previous_move,
            )

        self.assertEqual(
            write_map.call_args_list[0].kwargs["sa_previous_move"],
            previous_move,
        )
        self.assertIsNone(write_map.call_args_list[1].kwargs["sa_previous_move"])


if __name__ == "__main__":
    unittest.main()
