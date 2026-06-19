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

    def test_sa_game_starts_without_previous_move(self):
        response = self.make_game({"mode": "escape", "opponent_ai": "sa"})
        game_id = response.get_json()["game_id"]

        self.assertIsNone(main.games[game_id].sa_previous_move)

    def test_sa_move_records_and_reuses_previous_move(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "sa", "item_count": 0},
            spawns=((0, 0), (0, 5)),
        )
        game_id = response.get_json()["game_id"]

        with patch(
            "main.run_cpp_map_algorithm",
            side_effect=[((1, 0), (0, 3)), ((2, 0), (0, 1))],
        ) as run_algorithm:
            first = self.client.post(
                f"/api/games/{game_id}/move",
                json={"direction": "down"},
            )
            second = self.client.post(
                f"/api/games/{game_id}/move",
                json={"direction": "down"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(main.games[game_id].sa_previous_move, ((0, 3), (0, 1)))
        self.assertIsNone(run_algorithm.call_args_list[0].kwargs["sa_previous_move"])
        self.assertEqual(
            run_algorithm.call_args_list[1].kwargs["sa_previous_move"],
            ((0, 5), (0, 3)),
        )

    def test_sa_previous_move_is_not_updated_when_invisible(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "sa", "item_count": 0},
            spawns=((0, 0), (0, 5)),
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.items = [main.ItemState(type="invisibility_cloak", pos=(1, 0))]
        game.sa_previous_move = ((0, 7), (0, 5))

        with patch("main.run_cpp_map_algorithm") as run_algorithm:
            move = self.client.post(
                f"/api/games/{game_id}/move",
                json={"direction": "down"},
            )

        self.assertEqual(move.status_code, 200)
        run_algorithm.assert_not_called()
        self.assertEqual(game.sa_previous_move, ((0, 7), (0, 5)))

    def test_sa_previous_move_is_not_updated_when_monster_frozen(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "sa", "item_count": 0},
            spawns=((0, 0), (0, 5)),
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.monster_frozen_turns = [2]
        game.sa_previous_move = ((0, 7), (0, 5))

        with patch("main.run_cpp_map_algorithm") as run_algorithm:
            move = self.client.post(
                f"/api/games/{game_id}/move",
                json={"direction": "down"},
            )

        self.assertEqual(move.status_code, 200)
        run_algorithm.assert_not_called()
        self.assertEqual(game.sa_previous_move, ((0, 7), (0, 5)))

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

    def test_items_are_serialized_on_creation(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "item_count": 1}
        )

        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertEqual(len(data["items"]), 1)
        self.assertIn(data["items"][0]["type"], main.ITEM_KINDS)
        self.assertIn("effects", data)

    def test_speed_boots_grant_three_extra_direction_inputs(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "item_count": 0},
            spawns=((0, 0), (0, 10)),
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.items = [main.ItemState(type="speed_boots", pos=(1, 0))]

        move = self.client.post(f"/api/games/{game_id}/move", json={"direction": "down"})

        self.assertEqual(move.status_code, 200)
        data = move.get_json()
        self.assertEqual(data["human"], {"row": 1, "col": 0})
        self.assertEqual(data["monsters"], [{"row": 0, "col": 10}])
        self.assertEqual(data["effects"]["human_extra_steps"], 3)
        self.assertEqual(data["items"], [])

        second = self.client.post(
            f"/api/games/{game_id}/move", json={"direction": "right"}
        ).get_json()
        self.assertEqual(second["human"], {"row": 1, "col": 1})
        self.assertEqual(second["effects"]["human_extra_steps"], 2)
        self.assertEqual(second["monsters"], [{"row": 0, "col": 10}])

        self.client.post(
            f"/api/games/{game_id}/move", json={"direction": "down"}
        )
        fourth = self.client.post(
            f"/api/games/{game_id}/move", json={"direction": "left"}
        ).get_json()
        self.assertEqual(fourth["effects"]["human_extra_steps"], 0)
        self.assertEqual(fourth["monsters"], [{"row": 0, "col": 8}])

    def test_home_stone_teleports_to_a_safe_cell_and_ends_bonus_movement(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "item_count": 0},
            spawns=((0, 0), (0, 10)),
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.items = [main.ItemState(type="home_stone", pos=(1, 0))]

        with patch("main.random.choice", side_effect=lambda cells: cells[0]):
            move = self.client.post(
                f"/api/games/{game_id}/move", json={"direction": "down"}
            )

        self.assertEqual(move.status_code, 200)
        data = move.get_json()
        human = (data["human"]["row"], data["human"]["col"])
        self.assertNotEqual(human, (1, 0))
        self.assertGreaterEqual(
            main.distance_field(game.grid, [(0, 10)])[human[0]][human[1]],
            main.TELEPORT_SAFE_DISTANCE,
        )

    def test_freeze_trap_stops_monster_after_it_steps_on_trap(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "item_count": 0},
            spawns=((0, 0), (0, 5)),
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.traps = [
            main.ItemState(
                type="freeze_trap",
                pos=(0, 3),
                lifetime=main.FREEZE_TRAP_LIFETIME,
            )
        ]

        move = self.client.post(f"/api/games/{game_id}/move", json={"direction": "down"})

        self.assertEqual(move.status_code, 200)
        data = move.get_json()
        self.assertEqual(data["monsters"], [{"row": 0, "col": 3}])
        self.assertEqual(
            data["monster_states"][0]["frozen_turns"],
            main.FREEZE_TRAP_DURATION,
        )
        self.assertEqual(data["traps"], [])

    def test_invisibility_cloak_prevents_monster_move_for_turn(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "item_count": 0},
            spawns=((0, 0), (0, 5)),
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.items = [main.ItemState(type="invisibility_cloak", pos=(1, 0))]

        move = self.client.post(f"/api/games/{game_id}/move", json={"direction": "down"})

        self.assertEqual(move.status_code, 200)
        data = move.get_json()
        self.assertEqual(data["monsters"], [{"row": 0, "col": 5}])
        self.assertEqual(
            data["effects"]["human_invisible_turns"],
            main.INVISIBILITY_DURATION,
        )


if __name__ == "__main__":
    unittest.main()
