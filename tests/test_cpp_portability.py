from __future__ import annotations

import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
