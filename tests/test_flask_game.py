from __future__ import annotations

import unittest
from unittest.mock import patch

import main


def open_grid():
    return [[0 for _ in range(30)] for _ in range(30)]


def wall_grid():
    grid = open_grid()
    grid[0][1] = 1
    return grid


class FlaskGameTest(unittest.TestCase):
    def setUp(self):
        main.games.clear()
        self.client = main.app.test_client()

    def make_game(self, payload, grid=None, spawns=((0, 0), (0, 5))):
        with patch(
            "main.run_cpp_genetic_map",
            return_value=(grid or open_grid(), spawns[0], spawns[1]),
        ):
            return self.client.post("/api/games", json=payload)

    def test_create_escape_game(self):
        response = self.make_game({"mode": "escape", "opponent_ai": "astar"})

        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertEqual(len(data["grid"]), 30)
        self.assertEqual(len(data["grid"][0]), 30)
        self.assertEqual(data["mode"], "escape")
        self.assertEqual(data["opponent_ai"], "astar")
        self.assertEqual(len(data["monsters"]), 1)

    def test_create_chase_game(self):
        response = self.make_game({"mode": "chase"})

        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertEqual(data["mode"], "chase")
        self.assertEqual(data["opponent_ai"], "bfs")
        self.assertEqual(len(data["monsters"]), 1)

    def test_invalid_move_returns_400(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar"},
            grid=wall_grid(),
        )
        game_id = response.get_json()["game_id"]

        move = self.client.post(f"/api/games/{game_id}/move", json={"direction": "right"})

        self.assertEqual(move.status_code, 400)
        self.assertIn("wall", move.get_json()["error"])

    def test_escape_move_advances_human_and_monster(self):
        response = self.make_game({"mode": "escape", "opponent_ai": "astar"})
        game_id = response.get_json()["game_id"]

        move = self.client.post(f"/api/games/{game_id}/move", json={"direction": "down"})

        self.assertEqual(move.status_code, 200)
        data = move.get_json()
        self.assertEqual(data["human"], {"row": 1, "col": 0})
        self.assertEqual(data["monsters"], [{"row": 0, "col": 3}])
        self.assertEqual(data["step_count"], 1)

    def test_chase_move_renders_each_monster_step_before_bfs_human(self):
        response = self.make_game({"mode": "chase"})
        game_id = response.get_json()["game_id"]

        with patch("main.run_cpp_map_algorithm", return_value=((0, 29), (0, 3))):
            first = self.client.post(
                f"/api/games/{game_id}/move",
                json={"direction": "left"},
            )
            second = self.client.post(
                f"/api/games/{game_id}/move",
                json={"direction": "left"},
            )

        self.assertEqual(first.status_code, 200)
        first_data = first.get_json()
        self.assertEqual(first_data["monsters"], [{"row": 0, "col": 4}])
        self.assertEqual(first_data["human"], {"row": 0, "col": 0})
        self.assertEqual(first_data["pending_monster_steps"], 1)
        self.assertEqual(first_data["step_count"], 0)

        self.assertEqual(second.status_code, 200)
        data = second.get_json()
        self.assertEqual(data["monsters"], [{"row": 0, "col": 3}])
        self.assertEqual(data["human"], {"row": 0, "col": 29})
        self.assertEqual(data["pending_monster_steps"], 0)
        self.assertEqual(data["step_count"], 1)

    def test_astar_two_monster_creation(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "monster_count": 2}
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(len(response.get_json()["monsters"]), 2)

    def test_non_astar_forces_single_monster(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "greedy", "monster_count": 2}
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(len(response.get_json()["monsters"]), 1)

    def test_ended_game_rejects_more_moves(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar"},
            spawns=((0, 0), (0, 2)),
        )
        game_id = response.get_json()["game_id"]
        first = self.client.post(f"/api/games/{game_id}/move", json={"direction": "right"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.get_json()["status"], "ended")

        second = self.client.post(f"/api/games/{game_id}/move", json={"direction": "right"})

        self.assertEqual(second.status_code, 400)
        self.assertIn("ended", second.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
