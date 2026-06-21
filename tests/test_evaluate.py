from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


EVALUATE_PATH = Path(__file__).with_name("evaluate.py")
SPEC = importlib.util.spec_from_file_location("maze_evaluate", EVALUATE_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate
SPEC.loader.exec_module(evaluate)


class EvaluateTest(unittest.TestCase):
    def test_evaluate_excludes_uncaught_games_from_step_statistics(self):
        maps = [([[0]], (0, 0), (0, 0)) for _ in range(3)]
        results = [
            evaluate.GameResult(10, True, []),
            evaluate.GameResult(300, False, []),
            evaluate.GameResult(20, True, []),
        ]

        with patch.object(evaluate, "simulate", side_effect=results):
            stats = evaluate.evaluate("test", object(), maps)

        self.assertEqual(stats["steps_all"], [10.0, 20.0])
        self.assertEqual(stats["steps_mean"], 15.0)
        self.assertEqual(stats["steps_median"], 15.0)
        self.assertAlmostEqual(stats["caught_rate"], 2 / 3)

    def test_simulate_passes_previous_sa_move_to_next_turn(self):
        grid = [[0 for _ in range(30)] for _ in range(30)]

        with (
            patch.object(evaluate, "MAX_STEPS", 2),
            patch.object(evaluate, "player_move", return_value="down"),
            patch.object(
                evaluate,
                "run_cpp_map_algorithm",
                side_effect=[((11, 10), (0, 2)), ((12, 10), (0, 4))],
            ) as run_algorithm,
        ):
            evaluate.simulate(grid, (10, 10), [(0, 0)], evaluate.ai_sa)

        self.assertIsNone(
            run_algorithm.call_args_list[0].kwargs["sa_previous_move"]
        )
        self.assertEqual(
            run_algorithm.call_args_list[1].kwargs["sa_previous_move"],
            ((0, 0), (0, 2)),
        )


if __name__ == "__main__":
    unittest.main()
