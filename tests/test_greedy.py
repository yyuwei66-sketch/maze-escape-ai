from __future__ import annotations

import random
import unittest

from ai.greedy import make_greedy_controller


class GreedyMonsterAITest(unittest.TestCase):
    def test_random_walks_when_no_neighbor_can_get_closer(self):
        ctrl = make_greedy_controller(
            walls=[(0, 1), (1, 0)],
            width=5,
            height=5,
            steps_per_turn=1,
        )
        ctrl.rng = random.Random(0)

        path = ctrl.decide((0, 0), [(1, 1)])[0]

        self.assertEqual(path[0], (1, 1))
        self.assertIn(path[-1], {(2, 1), (1, 2)})
        self.assertNotEqual(path[-1], (1, 1))

    def test_blocked_manhattan_greedy_falls_back_to_bfs(self):
        ctrl = make_greedy_controller(
            walls=[(1, 2)],
            width=5,
            height=5,
            steps_per_turn=1,
        )

        first_path = ctrl.decide((0, 2), [(2, 2)])[0]
        second_path = ctrl.decide((0, 2), [first_path[-1]])[0]

        self.assertEqual(first_path, [(2, 2), (3, 2)])
        self.assertEqual(ctrl.bfs_fallback_steps(1), [5])
        self.assertEqual(second_path, [(3, 2), (4, 2)])


if __name__ == "__main__":
    unittest.main()
