from __future__ import annotations

from pathlib import Path
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
        main._generated_map_hashes.clear()
        self.client = main.app.test_client()

    def make_game(self, payload, grid=None, spawns=((0, 0), (0, 5))):
        with patch(
            "main.run_cpp_genetic_map",
            return_value=(grid or open_grid(), spawns[0], spawns[1]),
        ):
            return self.client.post("/api/games", json=payload)

    def assertMonsterPositions(self, monsters, expected):
        self.assertEqual(
            [{"row": monster["row"], "col": monster["col"]} for monster in monsters],
            expected,
        )

    def test_human_ui_has_swift_boots_end_turn_control(self):
        html = Path("ui/play/human.html").read_text(encoding="utf-8")

        self.assertIn("Swift Boots", html)
        self.assertIn("id=\"end-turn\"", html)
        self.assertIn("JSON.stringify({ action: 'end_turn' })", html)
        self.assertNotIn("dashRequested", html)

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

    def test_new_games_use_unique_seeds_maps_and_grid_objects(self):
        first_grid = open_grid()
        second_grid = open_grid()
        first_grid[1][1] = 1
        second_grid[2][2] = 1

        with (
            patch("main.secrets.randbits", side_effect=[101, 202]),
            patch(
                "main.run_cpp_genetic_map",
                side_effect=[
                    (first_grid, (0, 0), (0, 5)),
                    (second_grid, (0, 0), (0, 5)),
                ],
            ) as generator,
        ):
            first = self.client.post("/api/games", json={"mode": "chase"})
            second = self.client.post("/api/games", json={"mode": "chase"})

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        first_data = first.get_json()
        second_data = second.get_json()
        self.assertNotEqual(first_data["grid"], second_data["grid"])
        self.assertEqual(first_data["map_seed"], 101)
        self.assertEqual(second_data["map_seed"], 202)
        generator.assert_any_call(seed=101)
        generator.assert_any_call(seed=202)

        first_game = main.games[first_data["game_id"]]
        second_game = main.games[second_data["game_id"]]
        self.assertIsNot(first_game.grid, second_game.grid)
        self.assertIsNot(first_game.grid[0], second_game.grid[0])

    def test_duplicate_generated_layout_is_retried(self):
        duplicate_grid = open_grid()
        unique_grid = open_grid()
        unique_grid[3][3] = 1

        existing = main.GameState(
            grid=[list(row) for row in duplicate_grid],
            human=(0, 0),
            human_spawn=(0, 0),
            monsters=[(0, 5)],
            mode="chase",
            opponent_ai="bfs",
        )
        main.games[existing.game_id] = existing
        main._generated_map_hashes.add(main.map_layout_hash(existing.grid))

        with (
            patch("main.secrets.randbits", side_effect=[303, 404]),
            patch(
                "main.run_cpp_genetic_map",
                side_effect=[
                    (duplicate_grid, (0, 0), (0, 5)),
                    (unique_grid, (0, 0), (0, 5)),
                ],
            ) as generator,
        ):
            response = self.client.post("/api/games", json={"mode": "chase"})

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["map_seed"], 404)
        self.assertEqual(generator.call_count, 2)

    def test_invalid_move_returns_400(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar"},
            grid=wall_grid(),
        )
        game_id = response.get_json()["game_id"]

        move = self.client.post(f"/api/games/{game_id}/move", json={"direction": "right"})

        self.assertEqual(move.status_code, 400)
        self.assertIn("wall", move.get_json()["error"])

        game = main.games[game_id]
        self.assertEqual(game.step_count, 0)
        self.assertEqual(game.remaining_player_steps_this_turn, 0)
        self.assertFalse(game.waiting_for_second_step)

    def test_escape_move_advances_human_and_monster(self):
        response = self.make_game({"mode": "escape", "opponent_ai": "astar"})
        game_id = response.get_json()["game_id"]

        move = self.client.post(f"/api/games/{game_id}/move", json={"direction": "down"})

        self.assertEqual(move.status_code, 200)
        data = move.get_json()
        self.assertEqual(data["human"], {"row": 1, "col": 0})
        self.assertMonsterPositions(data["monsters"], [{"row": 0, "col": 3}])
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
        self.assertMonsterPositions(first_data["monsters"], [{"row": 0, "col": 4}])
        self.assertEqual(first_data["human"], {"row": 0, "col": 0})
        self.assertEqual(first_data["pending_monster_steps"], 1)
        self.assertEqual(first_data["step_count"], 0)

        self.assertEqual(second.status_code, 200)
        data = second.get_json()
        self.assertMonsterPositions(data["monsters"], [{"row": 0, "col": 3}])
        self.assertEqual(data["human"], {"row": 0, "col": 29})
        self.assertEqual(data["pending_monster_steps"], 0)
        self.assertEqual(data["step_count"], 1)

    def test_chase_bfs_reuses_previous_human_position(self):
        response = self.make_game({"mode": "chase"})
        game_id = response.get_json()["game_id"]

        with patch(
            "main.run_cpp_map_algorithm",
            side_effect=[((0, 29), (0, 3)), ((0, 28), (0, 1))],
        ) as run_algorithm:
            for _ in range(4):
                move = self.client.post(
                    f"/api/games/{game_id}/move",
                    json={"direction": "left"},
                )

        self.assertEqual(move.status_code, 200)
        self.assertIsNone(
            run_algorithm.call_args_list[0].kwargs["bfs_previous_human"]
        )
        self.assertEqual(
            run_algorithm.call_args_list[1].kwargs["bfs_previous_human"],
            (0, 0),
        )
        self.assertEqual(main.games[game_id].bfs_previous_human, (0, 29))

    def test_chase_bfs_can_backtrack_when_previous_cell_is_only_exit(self):
        grid = [[1 for _ in range(30)] for _ in range(30)]
        for row, col in ((1, 1), (1, 2), (1, 3)):
            grid[row][col] = 0

        game = main.GameState(
            grid=grid,
            human=(1, 1),
            human_spawn=(1, 1),
            monsters=[(1, 3)],
            mode="chase",
            opponent_ai="bfs",
            monster_frozen_turns=[0],
            bfs_previous_human=(1, 2),
        )

        moved_human = main.move_human_with_bfs(game)

        self.assertEqual(moved_human, (1, 2))
        self.assertEqual(game.bfs_previous_human, (1, 1))

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

    def test_sa_cpp_failure_falls_back_without_desyncing_human_move(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "sa", "item_count": 0},
            spawns=((0, 0), (0, 5)),
        )
        game_id = response.get_json()["game_id"]

        with patch(
            "main.run_cpp_map_algorithm",
            side_effect=main.CppAlgorithmError("SA failed"),
        ):
            move = self.client.post(
                f"/api/games/{game_id}/move",
                json={"direction": "down"},
            )

        self.assertEqual(move.status_code, 200)
        data = move.get_json()
        self.assertEqual(data["human"], {"row": 1, "col": 0})
        self.assertMonsterPositions(data["monsters"], [{"row": 0, "col": 3}])
        self.assertEqual(main.games[game_id].human, (1, 0))

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

    def test_speed_boots_allow_two_different_moves_before_monsters_move(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "item_count": 0},
            spawns=((0, 0), (0, 10)),
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.items = [main.ItemState(type="speed_boots", pos=(1, 0))]
        game.monster_frozen_turns = [99]

        move = self.client.post(f"/api/games/{game_id}/move", json={"direction": "down"})

        self.assertEqual(move.status_code, 200)
        data = move.get_json()
        self.assertEqual(data["human"], {"row": 1, "col": 0})
        self.assertMonsterPositions(data["monsters"], [{"row": 0, "col": 10}])
        self.assertEqual(data["effects"]["speed_boots_turns"], 5)
        self.assertEqual(data["items"], [])

        second = self.client.post(
            f"/api/games/{game_id}/move", json={"direction": "right"}
        ).get_json()
        self.assertEqual(second["human"], {"row": 1, "col": 1})
        self.assertEqual(second["effects"]["speed_boots_turns"], 5)
        self.assertTrue(second["waitingForSecondStep"])
        self.assertEqual(second["remainingPlayerStepsThisTurn"], 1)
        self.assertMonsterPositions(second["monsters"], [{"row": 0, "col": 10}])

        completed = self.client.post(
            f"/api/games/{game_id}/move",
            json={"direction": "down"},
        ).get_json()
        self.assertEqual(completed["human"], {"row": 2, "col": 1})
        self.assertEqual(completed["effects"]["speed_boots_turns"], 4)
        self.assertFalse(completed["waitingForSecondStep"])

    def test_blocked_second_boots_step_finishes_turn(self):
        grid = open_grid()
        grid[0][2] = 1
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "item_count": 0},
            grid=grid,
            spawns=((0, 0), (0, 10)),
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.human_speed_boots_turns = 2
        game.monster_frozen_turns = [99]

        first = self.client.post(
            f"/api/games/{game_id}/move",
            json={"direction": "right"},
        )
        self.assertEqual(first.status_code, 200)
        self.assertTrue(first.get_json()["waitingForSecondStep"])

        second = self.client.post(
            f"/api/games/{game_id}/move", json={"direction": "right"}
        )
        self.assertEqual(second.status_code, 200)
        data = second.get_json()
        self.assertEqual(data["human"], {"row": 0, "col": 1})
        self.assertEqual(data["effects"]["speed_boots_turns"], 1)
        self.assertFalse(data["waitingForSecondStep"])

    def test_end_turn_skips_second_boots_step_and_moves_monsters(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "item_count": 0},
            spawns=((0, 0), (0, 10)),
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.human_speed_boots_turns = 2

        first = self.client.post(
            f"/api/games/{game_id}/move", json={"direction": "down"}
        ).get_json()
        self.assertTrue(first["waitingForSecondStep"])
        self.assertTrue(first["canEndTurn"])
        self.assertMonsterPositions(first["monsters"], [{"row": 0, "col": 10}])

        ended = self.client.post(
            f"/api/games/{game_id}/move", json={"action": "end_turn"}
        ).get_json()
        self.assertFalse(ended["waitingForSecondStep"])
        self.assertFalse(ended["canEndTurn"])
        self.assertEqual(ended["speedBootsTurns"], 1)
        self.assertMonsterPositions(ended["monsters"], [{"row": 0, "col": 8}])

    def test_repick_boots_refreshes_without_decaying_same_turn(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "item_count": 0},
            spawns=((0, 0), (0, 10)),
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.human_speed_boots_turns = 1
        game.monster_frozen_turns = [99]
        game.items = [main.ItemState(type="speed_boots", pos=(1, 0))]

        self.client.post(
            f"/api/games/{game_id}/move", json={"direction": "down"}
        )
        completed = self.client.post(
            f"/api/games/{game_id}/move", json={"direction": "right"}
        ).get_json()

        self.assertEqual(completed["speedBootsTurns"], main.SPEED_BOOTS_DURATION)

    def test_item_pickup_is_checked_after_first_boots_step(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "item_count": 0},
            spawns=((0, 0), (0, 10)),
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.human_speed_boots_turns = 2
        game.monster_frozen_turns = [99]
        game.items = [main.ItemState(type="invisibility_cloak", pos=(0, 1))]

        data = self.client.post(
            f"/api/games/{game_id}/move", json={"direction": "right"}
        ).get_json()

        self.assertEqual(data["human"], {"row": 0, "col": 1})
        self.assertEqual(data["invisibleTurns"], main.INVISIBILITY_DURATION)
        self.assertEqual(data["pickedItems"], ["invisibility_cloak"])
        self.assertTrue(data["waitingForSecondStep"])
        self.assertIn("choose your second move", data["message"])

    def test_freeze_pickup_on_first_boots_step_still_allows_second_step(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "item_count": 0},
            spawns=((0, 0), (0, 10)),
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.human_speed_boots_turns = 2
        game.items = [main.ItemState(type="freeze_trap", pos=(1, 0))]

        first = self.client.post(
            f"/api/games/{game_id}/move", json={"direction": "down"}
        ).get_json()
        self.assertTrue(first["waitingForSecondStep"])
        self.assertEqual(first["monster_states"][0]["frozen_turns"], 5)

        second = self.client.post(
            f"/api/games/{game_id}/move", json={"direction": "right"}
        ).get_json()
        self.assertEqual(second["human"], {"row": 1, "col": 1})
        self.assertFalse(second["waitingForSecondStep"])
        self.assertEqual(second["monster_states"][0]["frozen_turns"], 5)

    def test_state_exposes_player_turn_and_canonical_effect_fields(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "item_count": 0}
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.human_speed_boots_turns = 3
        game.human_invisible_turns = 2

        data = self.client.get(f"/api/games/{game_id}").get_json()

        self.assertEqual(data["speedBootsTurns"], 3)
        self.assertEqual(data["invisibleTurns"], 2)
        self.assertEqual(data["remainingPlayerStepsThisTurn"], 0)
        self.assertEqual(data["maxPlayerStepsThisTurn"], 1)
        self.assertFalse(data["waitingForSecondStep"])
        self.assertFalse(data["canEndTurn"])
        self.assertFalse(data["gameOver"])
        self.assertIsNone(data["outcome"])
        self.assertFalse(data["won"])
        self.assertFalse(data["lost"])
        self.assertIn("frozenTurns", data["monsters"][0])
        self.assertNotIn("human_extra_steps", data["effects"])

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

    def test_home_stone_interrupts_first_boots_step(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "item_count": 0},
            spawns=((0, 0), (0, 10)),
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.human_speed_boots_turns = 2
        game.monster_frozen_turns = [99]
        game.items = [main.ItemState(type="home_stone", pos=(1, 0))]

        with patch("main.random.choice", side_effect=lambda cells: cells[0]):
            data = self.client.post(
                f"/api/games/{game_id}/move", json={"direction": "down"}
            ).get_json()

        self.assertFalse(data["waitingForSecondStep"])
        self.assertEqual(data["remainingPlayerStepsThisTurn"], 0)
        self.assertEqual(data["step_count"], 1)

    def test_safe_teleport_excludes_monsters_items_and_traps(self):
        game = main.GameState(
            grid=open_grid(),
            human=(0, 0),
            human_spawn=(0, 0),
            monsters=[(5, 5), (20, 20)],
            mode="escape",
            opponent_ai="astar",
            monster_frozen_turns=[0, 0],
            items=[main.ItemState(type="speed_boots", pos=(10, 10))],
            traps=[main.ItemState(type="freeze_trap", pos=(15, 15))],
        )

        with patch("main.random.choice", side_effect=lambda cells: cells[0]):
            target = main.find_safe_teleport_position(game)

        self.assertNotEqual(target, game.human)
        self.assertNotIn(target, game.monsters)
        self.assertNotEqual(target, (10, 10))
        self.assertNotEqual(target, (15, 15))
        self.assertEqual(game.grid[target[0]][target[1]], 0)

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
        self.assertMonsterPositions(data["monsters"], [{"row": 0, "col": 3}])
        self.assertEqual(
            data["monster_states"][0]["frozen_turns"],
            main.FREEZE_TRAP_DURATION,
        )
        self.assertEqual(data["traps"], [])

    def test_freeze_item_freezes_monster_immediately_when_picked_up(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "item_count": 0},
            spawns=((0, 0), (0, 5)),
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.items = [main.ItemState(type="freeze_trap", pos=(1, 0))]

        move = self.client.post(f"/api/games/{game_id}/move", json={"direction": "down"})

        self.assertEqual(move.status_code, 200)
        data = move.get_json()
        self.assertMonsterPositions(data["monsters"], [{"row": 0, "col": 5}])
        self.assertEqual(
            data["monster_states"][0]["frozen_turns"],
            main.FREEZE_TRAP_DURATION,
        )
        self.assertEqual(data["items"], [])
        self.assertEqual(data["traps"], [])

    def test_frost_duration_decays_once_after_complete_boots_turn(self):
        game = main.GameState(
            grid=open_grid(),
            human=(0, 0),
            human_spawn=(0, 0),
            monsters=[(0, 10)],
            mode="escape",
            opponent_ai="astar",
            human_speed_boots_turns=2,
            monster_frozen_turns=[4],
        )

        main.apply_player_move(game, {"direction": "down"})
        self.assertTrue(game.waiting_for_second_step)
        self.assertEqual(game.monster_frozen_turns, [4])

        main.apply_player_move(game, {"direction": "right"})
        self.assertFalse(game.waiting_for_second_step)
        self.assertEqual(game.monster_frozen_turns, [3])

    def test_repick_frost_preserves_larger_duration_without_stacking(self):
        game = main.GameState(
            grid=open_grid(),
            human=(0, 0),
            human_spawn=(0, 0),
            monsters=[(0, 5), (5, 0)],
            mode="escape",
            opponent_ai="astar",
            monster_frozen_turns=[2, 8],
        )

        frost = main.ItemState(type="freeze_trap", pos=(0, 0))
        main.apply_item_effect(game, frost)

        self.assertEqual(game.monster_frozen_turns, [5, 8])

    def test_invisibility_cloak_makes_monster_move_randomly_without_capture(self):
        response = self.make_game(
            {"mode": "escape", "opponent_ai": "astar", "item_count": 0},
            spawns=((0, 0), (0, 5)),
        )
        game_id = response.get_json()["game_id"]
        game = main.games[game_id]
        game.items = [main.ItemState(type="invisibility_cloak", pos=(1, 0))]

        with patch("main.random.choice", side_effect=lambda cells: cells[0]):
            move = self.client.post(
                f"/api/games/{game_id}/move", json={"direction": "down"}
            )

        self.assertEqual(move.status_code, 200)
        data = move.get_json()
        self.assertMonsterPositions(data["monsters"], [{"row": 28, "col": 5}])
        self.assertEqual(
            data["effects"]["human_invisible_turns"],
            main.INVISIBILITY_DURATION,
        )
        self.assertEqual(data["invisibleTurns"], 10)
        self.assertTrue(data["isInvisible"])
        self.assertIn("Cloak activated: invisible for 10 turns.", data["message"])

    def test_cloak_decays_after_one_normal_player_turn(self):
        game = main.GameState(
            grid=open_grid(),
            human=(0, 0),
            human_spawn=(0, 0),
            monsters=[(0, 10)],
            mode="escape",
            opponent_ai="astar",
            human_invisible_turns=4,
            monster_frozen_turns=[0],
        )

        with patch("main.random.choice", side_effect=lambda cells: cells[0]):
            main.apply_player_move(game, {"direction": "down"})

        self.assertEqual(game.human_invisible_turns, 3)

    def test_invisible_collision_does_not_end_game(self):
        game = main.GameState(
            grid=open_grid(),
            human=(0, 0),
            human_spawn=(0, 0),
            monsters=[(0, 0)],
            mode="escape",
            opponent_ai="astar",
            human_invisible_turns=3,
            monster_frozen_turns=[0],
        )

        self.assertFalse(main.catches_human(game))
        main.update_status(game)
        self.assertEqual(game.status, "running")

    def test_invisible_monsters_bypass_chasing_logic(self):
        game = main.GameState(
            grid=open_grid(),
            human=(0, 0),
            human_spawn=(0, 0),
            monsters=[(0, 5)],
            mode="escape",
            opponent_ai="astar",
            human_invisible_turns=3,
            monster_frozen_turns=[0],
        )

        with patch("main.planned_monster_paths") as chase, patch(
            "main.random.choice", side_effect=lambda cells: cells[0]
        ):
            moved = main.move_monsters_with_item_effects(game)

        chase.assert_not_called()
        self.assertNotEqual(moved, [(0, 5)])
        self.assertFalse(main.catches_human(game))

    def test_cloak_decays_once_after_complete_boots_turn(self):
        game = main.GameState(
            grid=open_grid(),
            human=(0, 0),
            human_spawn=(0, 0),
            monsters=[(0, 10)],
            mode="escape",
            opponent_ai="astar",
            human_speed_boots_turns=2,
            human_invisible_turns=4,
            monster_frozen_turns=[0],
        )

        main.apply_player_move(game, {"direction": "down"})
        self.assertTrue(game.waiting_for_second_step)
        self.assertEqual(game.human_invisible_turns, 4)

        with patch("main.random.choice", side_effect=lambda cells: cells[0]):
            main.apply_player_move(game, {"direction": "right"})
        self.assertFalse(game.waiting_for_second_step)
        self.assertEqual(game.human_invisible_turns, 3)

    def test_repick_cloak_refreshes_without_stacking_or_shortening(self):
        game = main.GameState(
            grid=open_grid(),
            human=(0, 0),
            human_spawn=(0, 0),
            monsters=[(0, 5)],
            mode="escape",
            opponent_ai="astar",
            human_invisible_turns=4,
            monster_frozen_turns=[0],
        )
        cloak = main.ItemState(type="invisibility_cloak", pos=(0, 0))
        main.apply_item_effect(game, cloak)
        self.assertEqual(game.human_invisible_turns, 10)

        game.human_invisible_turns = 12
        second_cloak = main.ItemState(type="invisibility_cloak", pos=(0, 0))
        main.apply_item_effect(game, second_cloak)
        self.assertEqual(game.human_invisible_turns, 12)

    def test_capture_resumes_after_invisibility_expires(self):
        game = main.GameState(
            grid=open_grid(),
            human=(0, 0),
            human_spawn=(0, 0),
            monsters=[(0, 0)],
            mode="escape",
            opponent_ai="astar",
            human_invisible_turns=0,
            monster_frozen_turns=[0],
        )

        self.assertTrue(main.catches_human(game))
        main.update_status(game)
        self.assertEqual(game.status, "ended")

    def test_dual_monsters_wander_while_player_is_invisible(self):
        game = main.GameState(
            grid=open_grid(),
            human=(0, 0),
            human_spawn=(0, 0),
            monsters=[(0, 5), (5, 0)],
            mode="escape",
            opponent_ai="astar",
            human_invisible_turns=3,
            monster_frozen_turns=[0, 0],
        )

        with patch("main.planned_monster_paths") as chase, patch(
            "main.random.choice", side_effect=lambda cells: cells[0]
        ):
            moved = main.move_monsters_with_item_effects(game)

        chase.assert_not_called()
        self.assertNotEqual(moved[0], (0, 5))
        self.assertNotEqual(moved[1], (5, 0))
        self.assertFalse(main.catches_human(game))

    def test_dual_monster_prediction_resumes_while_player_is_visible(self):
        game = main.GameState(
            grid=open_grid(),
            human=(0, 0),
            human_spawn=(0, 0),
            monsters=[(0, 5), (15, 15)],
            mode="escape",
            opponent_ai="astar",
            human_invisible_turns=0,
            monster_frozen_turns=[0, 0],
        )

        with patch(
            "main.get_prediction_intercept_target", return_value=(10, 10)
        ) as predictor:
            paths = main.planned_astar_paths(game)

        predictor.assert_called_once()
        self.assertEqual(len(paths), 2)

    def test_dual_monsters_both_respect_freeze(self):
        game = main.GameState(
            grid=open_grid(),
            human=(0, 0),
            human_spawn=(0, 0),
            monsters=[(0, 5), (5, 0)],
            mode="escape",
            opponent_ai="astar",
            monster_frozen_turns=[3, 3],
        )

        moved = main.move_monsters_with_item_effects(game)

        self.assertEqual(moved, [(0, 5), (5, 0)])
        self.assertEqual(game.last_monster_paths, [[(0, 5)], [(5, 0)]])


if __name__ == "__main__":
    unittest.main()
