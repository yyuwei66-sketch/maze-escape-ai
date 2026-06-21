from __future__ import annotations

import unittest

from ai.astar import heuristic


class AstarTest(unittest.TestCase):
    def test_heuristic_wraps_across_grid_edges(self):
        self.assertEqual(heuristic((0, 10), (29, 10)), 1)
        self.assertEqual(heuristic((10, 0), (10, 29)), 1)


if __name__ == "__main__":
    unittest.main()
