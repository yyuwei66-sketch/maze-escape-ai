from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import random
from typing import Any, Dict, List, Sequence, Tuple
import uuid

from flask import Flask, jsonify, request

from ai import (
    CppAlgorithmError,
    astar,
    make_greedy_controller,
    make_minimax_controller,
    many_row_col_to_xy,
    many_xy_to_row_col,
    run_cpp_genetic_map,
    run_cpp_map_algorithm,
    walls_from_grid,
)

Pos = Tuple[int, int]
ItemKind = str

VALID_DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ESCAPE_AIS = {"astar", "greedy", "minimax", "sa"}
CHASE_AI = "bfs"
ITEM_KINDS: tuple[ItemKind, ...] = (
    "speed_boots",
    "home_stone",
    "freeze_trap",
    "invisibility_cloak",
)
ITEM_LABELS = {
    "speed_boots": "Speed Boots",
    "home_stone": "Safe Teleport Stone",
    "freeze_trap": "Freeze Trap",
    "invisibility_cloak": "Invisibility Cloak",
}
ITEM_SYMBOLS = {
    "speed_boots": "S",
    "home_stone": "T",
    "freeze_trap": "F",
    "invisibility_cloak": "C",
}
ITEM_SPAWN_COUNT = 2
ITEM_SPAWN_INTERVAL = 10
SPEED_BOOTS_EXTRA_STEPS = 3
TELEPORT_SAFE_DISTANCE = 6
FREEZE_TRAP_LIFETIME = 20
FREEZE_TRAP_DURATION = 3
INVISIBILITY_DURATION = 4

app = Flask(__name__)
games: Dict[str, "GameState"] = {}


@dataclass
class ItemState:
    type: ItemKind
    pos: Pos
    active: bool = True
    lifetime: int = -1

    def serialize(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": ITEM_LABELS.get(self.type, self.type),
            "symbol": ITEM_SYMBOLS.get(self.type, "?"),
            "pos": pos_to_json(self.pos),
            "active": self.active,
            "lifetime": self.lifetime,
        }


@dataclass
class GameState:
    grid: List[List[int]]
    human: Pos
    human_spawn: Pos
    monsters: List[Pos]
    mode: str
    opponent_ai: str
    step_count: int = 0
    pending_monster_steps: int = 0
    status: str = "running"
    message: str = ""
    items: List[ItemState] = field(default_factory=list)
    traps: List[ItemState] = field(default_factory=list)
    cloak_already_spawned: bool = False
    human_extra_steps: int = 0
    human_invisible_turns: int = 0
    monster_frozen_turns: List[int] = field(default_factory=list)
    game_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def serialize(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "grid": self.grid,
            "human": pos_to_json(self.human),
            "monsters": [pos_to_json(pos) for pos in self.monsters],
            "monster_states": [
                {"pos": pos_to_json(pos), "frozen_turns": frozen}
                for pos, frozen in zip(self.monsters, self.monster_frozen_turns)
            ],
            "items": [item.serialize() for item in self.items if item.active],
            "traps": [trap.serialize() for trap in self.traps if trap.active],
            "effects": {
                "human_invisible_turns": self.human_invisible_turns,
                "human_extra_steps": self.human_extra_steps,
                "monster_frozen_turns": list(self.monster_frozen_turns),
            },
            "mode": self.mode,
            "opponent_ai": self.opponent_ai,
            "step_count": self.step_count,
            "pending_monster_steps": self.pending_monster_steps,
            "status": self.status,
            "message": self.message,
            "height": len(self.grid),
            "width": len(self.grid[0]) if self.grid else 0,
        }


def pos_to_json(pos: Pos) -> dict[str, int]:
    return {"row": pos[0], "col": pos[1]}


def json_error(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


def wrap_pos(pos: Pos, grid: Sequence[Sequence[int]]) -> Pos:
    return pos[0] % len(grid), pos[1] % len(grid[0])


def is_floor(grid: Sequence[Sequence[int]], pos: Pos) -> bool:
    row, col = wrap_pos(pos, grid)
    return int(grid[row][col]) == 0


def move_one(grid: Sequence[Sequence[int]], pos: Pos, direction: str) -> Pos:
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"invalid direction: {direction}")
    dr, dc = VALID_DIRECTIONS[direction]
    nxt = wrap_pos((pos[0] + dr, pos[1] + dc), grid)
    if not is_floor(grid, nxt):
        raise ValueError("move would hit a wall")
    return nxt


def assert_running(game: GameState) -> None:
    if game.status != "running":
        raise ValueError("game has already ended")


def update_status(game: GameState) -> None:
    if any(monster == game.human for monster in game.monsters):
        game.status = "ended"
        game.message = f"caught after {game.step_count} turns"


def create_game(payload: dict[str, Any]) -> GameState:
    mode = str(payload.get("mode", "escape")).lower()
    if mode not in {"escape", "chase"}:
        raise ValueError("mode must be 'escape' or 'chase'")

    if mode == "escape":
        opponent_ai = str(payload.get("opponent_ai", "astar")).lower()
        if opponent_ai not in ESCAPE_AIS:
            raise ValueError("escape opponent_ai must be astar, greedy, minimax, or sa")
        monster_count = int(payload.get("monster_count", 1))
        if opponent_ai == "astar":
            if monster_count not in {1, 2}:
                raise ValueError("astar monster_count must be 1 or 2")
        else:
            monster_count = 1
    else:
        opponent_ai = CHASE_AI
        monster_count = 1

    grid, human, first_monster = run_cpp_genetic_map()
    monsters = [first_monster]
    while len(monsters) < monster_count:
        extra = find_extra_monster_spawn(grid, human, monsters)
        if extra is None:
            raise ValueError("could not place the second monster")
        monsters.append(extra)

    game = GameState(
        grid=[list(row) for row in grid],
        human=human,
        human_spawn=human,
        monsters=monsters,
        mode=mode,
        opponent_ai=opponent_ai,
        monster_frozen_turns=[0 for _ in monsters],
    )
    if mode == "escape":
        spawn_random_items(game, int(payload.get("item_count", ITEM_SPAWN_COUNT)))
    update_status(game)
    games[game.game_id] = game
    return game


def find_extra_monster_spawn(
    grid: Sequence[Sequence[int]],
    human: Pos,
    monsters: Sequence[Pos],
) -> Pos | None:
    candidates = [
        (row, col)
        for row, cells in enumerate(grid)
        for col, value in enumerate(cells)
        if int(value) == 0 and (row, col) != human and (row, col) not in monsters
    ]
    candidates.sort(
        key=lambda pos: min(torus_manhattan(pos, human, grid), 999),
        reverse=True,
    )
    return candidates[0] if candidates else None


def torus_manhattan(a: Pos, b: Pos, grid: Sequence[Sequence[int]]) -> int:
    h = len(grid)
    w = len(grid[0])
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return min(dr, h - dr) + min(dc, w - dc)


def distance_field(
    grid: Sequence[Sequence[int]],
    sources: Sequence[Pos],
) -> List[List[int | None]]:
    distances: List[List[int | None]] = [
        [None for _ in row] for row in grid
    ]
    queue = deque()
    for source in sources:
        row, col = wrap_pos(source, grid)
        if not is_floor(grid, (row, col)) or distances[row][col] is not None:
            continue
        distances[row][col] = 0
        queue.append((row, col))

    while queue:
        row, col = queue.popleft()
        current = distances[row][col]
        assert current is not None
        for dr, dc in VALID_DIRECTIONS.values():
            neighbor = wrap_pos((row + dr, col + dc), grid)
            nr, nc = neighbor
            if not is_floor(grid, neighbor) or distances[nr][nc] is not None:
                continue
            distances[nr][nc] = current + 1
            queue.append(neighbor)
    return distances


def find_safe_teleport_position(
    game: GameState,
    min_distance: int = TELEPORT_SAFE_DISTANCE,
) -> Pos:
    distances = distance_field(game.grid, game.monsters)
    candidates = [
        (row, col)
        for row, cells in enumerate(game.grid)
        for col, value in enumerate(cells)
        if int(value) == 0
        and (row, col) != game.human
        and distances[row][col] is not None
        and distances[row][col] >= min_distance
    ]
    if candidates:
        return random.choice(candidates)

    fallback = [
        (distances[row][col], (row, col))
        for row, cells in enumerate(game.grid)
        for col, value in enumerate(cells)
        if int(value) == 0
        and (row, col) != game.human
        and distances[row][col] is not None
    ]
    return max(
        fallback,
        default=(None, game.human),
        key=lambda entry: entry[0] or -1,
    )[1]


def random_item_type(cloak_already_spawned: bool) -> ItemKind:
    total_weight = 90 if cloak_already_spawned else 95
    roll = random.randint(1, total_weight)
    if not cloak_already_spawned and roll <= 5:
        return "invisibility_cloak"

    current_offset = 0 if cloak_already_spawned else 5
    if roll <= current_offset + 20:
        return "home_stone"
    if roll <= current_offset + 20 + 35:
        return "freeze_trap"
    return "speed_boots"


def can_spawn_item(game: GameState, pos: Pos) -> bool:
    pos = wrap_pos(pos, game.grid)
    if not is_floor(game.grid, pos):
        return False
    if pos == game.human:
        return False
    if torus_manhattan(pos, game.human, game.grid) <= 1:
        return False
    if pos in game.monsters:
        return False
    if any(item.active and item.pos == pos for item in game.items):
        return False
    if any(trap.active and trap.pos == pos for trap in game.traps):
        return False
    return True


def spawn_random_items(game: GameState, item_count: int = ITEM_SPAWN_COUNT) -> None:
    if item_count <= 0 or not game.grid:
        return

    rows = len(game.grid)
    cols = len(game.grid[0])
    spawned = 0
    attempts = 0
    while spawned < item_count and attempts < 500:
        attempts += 1
        pos = (random.randint(0, rows - 1), random.randint(0, cols - 1))
        if not can_spawn_item(game, pos):
            continue

        item_type = random_item_type(game.cloak_already_spawned)
        if item_type == "invisibility_cloak":
            game.cloak_already_spawned = True
        game.items.append(
            ItemState(
                type=item_type,
                pos=pos,
                lifetime=FREEZE_TRAP_LIFETIME if item_type == "freeze_trap" else -1,
            )
        )
        spawned += 1


def update_trap_lifetimes(game: GameState) -> None:
    for trap in game.traps:
        if trap.active and trap.lifetime > 0:
            trap.lifetime -= 1
            if trap.lifetime <= 0:
                trap.active = False


def decay_effects_after_human_step(game: GameState) -> None:
    update_trap_lifetimes(game)
    if game.human_invisible_turns > 0:
        game.human_invisible_turns -= 1
    game.monster_frozen_turns = [
        max(0, frozen - 1) for frozen in game.monster_frozen_turns
    ]


def apply_item_effect(game: GameState, item: ItemState) -> bool:
    """Apply a picked-up item. Return True when movement should stop."""

    if item.type == "speed_boots":
        game.human_extra_steps += SPEED_BOOTS_EXTRA_STEPS
        item.active = False
        return False
    if item.type == "home_stone":
        game.human = find_safe_teleport_position(game)
        game.human_extra_steps = 0
        item.active = False
        return True
    if item.type == "freeze_trap":
        game.traps.append(
            ItemState(
                type="freeze_trap",
                pos=item.pos,
                active=True,
                lifetime=FREEZE_TRAP_LIFETIME,
            )
        )
        item.active = False
        return False
    if item.type == "invisibility_cloak":
        game.human_invisible_turns = INVISIBILITY_DURATION
        game.cloak_already_spawned = True
        item.active = False
        return False
    return False


def check_player_item_pickup(game: GameState) -> bool:
    for item in game.items:
        if item.active and item.pos == game.human:
            return apply_item_effect(game, item)
    return False


def move_human_with_items(game: GameState, direction: str) -> bool:
    using_extra_step = game.human_extra_steps > 0
    game.human = move_one(game.grid, game.human, direction)
    if using_extra_step:
        game.human_extra_steps -= 1

    decay_effects_after_human_step(game)
    stop_movement = check_player_item_pickup(game)
    if stop_movement:
        game.human_extra_steps = 0
    return stop_movement


def apply_player_move(game: GameState, payload: dict[str, Any]) -> None:
    assert_running(game)
    if game.mode == "escape":
        direction = str(payload.get("direction", "")).lower()
        movement_ended = move_human_with_items(game, direction)
        if (
            not catches_human(game)
            and (movement_ended or game.human_extra_steps == 0)
        ):
            game.monsters = move_monsters_with_item_effects(game)
    else:
        direction = str(payload.get("direction", "")).lower()
        game.monsters[0] = move_one(game.grid, game.monsters[0], direction)
        game.pending_monster_steps += 1
        if catches_human(game):
            game.step_count += 1
            game.pending_monster_steps = 0
        elif game.pending_monster_steps >= 2:
            game.human = move_human_with_bfs(game)
            game.step_count += 1
            game.pending_monster_steps = 0
        update_status(game)
        return

    game.step_count += 1
    if game.mode == "escape" and game.step_count % ITEM_SPAWN_INTERVAL == 0:
        spawn_random_items(game, ITEM_SPAWN_COUNT)
    update_status(game)


def catches_human(game: GameState) -> bool:
    return any(monster == game.human for monster in game.monsters)


def check_monster_trap(game: GameState, monster_index: int) -> bool:
    for trap in game.traps:
        if trap.active and trap.pos == game.monsters[monster_index]:
            game.monster_frozen_turns[monster_index] = FREEZE_TRAP_DURATION
            trap.active = False
            return True
    return False


def move_monsters_with_item_effects(game: GameState) -> List[Pos]:
    while len(game.monster_frozen_turns) < len(game.monsters):
        game.monster_frozen_turns.append(0)

    if game.human_invisible_turns > 0:
        return list(game.monsters)

    paths = planned_monster_paths(game)
    next_monsters = list(game.monsters)
    for index, path in enumerate(paths):
        if index >= len(next_monsters):
            break
        if game.monster_frozen_turns[index] > 0:
            continue

        for pos in path[1:]:
            next_monsters[index] = pos
            game.monsters[index] = pos
            if check_monster_trap(game, index):
                break
            if pos == game.human:
                break
    return next_monsters


def planned_monster_paths(game: GameState) -> List[List[Pos]]:
    ai_name = game.opponent_ai
    if ai_name == "astar":
        return planned_astar_paths(game)
    if ai_name in {"greedy", "minimax"}:
        return planned_controller_paths(game, ai_name)
    if ai_name == "sa":
        _, monster = run_cpp_map_algorithm(
            "sa",
            game.grid,
            game.human,
            game.monsters[0],
        )
        return [[game.monsters[0], monster]]
    raise ValueError(f"unsupported monster AI: {ai_name}")


def planned_astar_paths(game: GameState) -> List[List[Pos]]:
    if len(game.monsters) == 1:
        return [limited_path(game.grid, game.monsters[0], game.human, 2)]

    m1, m2 = game.monsters[:2]
    d1 = torus_manhattan(m1, game.human, game.grid)
    d2 = torus_manhattan(m2, game.human, game.grid)
    if d1 <= d2:
        return [
            limited_path(game.grid, m1, game.human, 2),
            limited_path(game.grid, m2, intercept_point(game.grid, m1, game.human), 1),
        ]
    return [
        limited_path(game.grid, m1, intercept_point(game.grid, m2, game.human), 1),
        limited_path(game.grid, m2, game.human, 2),
    ]


def limited_path(
    grid: Sequence[Sequence[int]],
    start: Pos,
    goal: Pos,
    steps: int,
) -> List[Pos]:
    path = astar([list(row) for row in grid], start, goal)
    if not path:
        return [start]
    return path[: steps + 1]


def planned_controller_paths(game: GameState, controller_name: str) -> List[List[Pos]]:
    walls = walls_from_grid(game.grid)
    width = len(game.grid[0])
    height = len(game.grid)
    human_xy = (game.human[1], game.human[0])
    monsters_xy = many_row_col_to_xy(game.monsters)
    if controller_name == "greedy":
        ctrl = make_greedy_controller(
            walls,
            width=width,
            height=height,
            steps_per_turn=2,
            allow_stay=False,
            avoid_stacking=True,
        )
    else:
        ctrl = make_minimax_controller(
            walls,
            width=width,
            height=height,
            steps_per_turn=2,
            depth=2,
            player_can_stay=False,
            monster_can_stay=False,
        )
    paths = ctrl.decide(human_xy, monsters_xy)
    return [many_xy_to_row_col(path) for path in paths]


def move_monsters(game: GameState) -> List[Pos]:
    ai_name = game.opponent_ai
    if ai_name == "astar":
        return move_monsters_astar(game)
    if ai_name == "greedy":
        return move_monsters_controller(game, "greedy")
    if ai_name == "minimax":
        return move_monsters_controller(game, "minimax")
    if ai_name == "sa":
        _, monster = run_cpp_map_algorithm(
            "sa",
            game.grid,
            game.human,
            game.monsters[0],
        )
        return [monster]
    raise ValueError(f"unsupported monster AI: {ai_name}")


def move_monsters_astar(game: GameState) -> List[Pos]:
    if len(game.monsters) == 1:
        return [advance_along_path(game.grid, game.monsters[0], game.human, 2)]

    m1, m2 = game.monsters[:2]
    d1 = torus_manhattan(m1, game.human, game.grid)
    d2 = torus_manhattan(m2, game.human, game.grid)
    if d1 <= d2:
        return [
            advance_along_path(game.grid, m1, game.human, 2),
            advance_along_path(game.grid, m2, intercept_point(game.grid, m1, game.human), 1),
        ]
    return [
        advance_along_path(game.grid, m1, intercept_point(game.grid, m2, game.human), 1),
        advance_along_path(game.grid, m2, game.human, 2),
    ]


def advance_along_path(
    grid: Sequence[Sequence[int]],
    start: Pos,
    goal: Pos,
    steps: int,
) -> Pos:
    path = astar([list(row) for row in grid], start, goal)
    if not path:
        return start
    return path[min(steps, len(path) - 1)]


def intercept_point(
    grid: Sequence[Sequence[int]],
    monster: Pos,
    human: Pos,
    offset_steps: int = 4,
) -> Pos:
    h = len(grid)
    w = len(grid[0])
    dr = human[0] - monster[0]
    dc = human[1] - monster[1]
    if abs(dr) > h // 2:
        dr = dr - h if dr > 0 else dr + h
    if abs(dc) > w // 2:
        dc = dc - w if dc > 0 else dc + w
    sign_r = (1 if dr > 0 else -1) if dr else 0
    sign_c = (1 if dc > 0 else -1) if dc else 0
    target = wrap_pos(
        (human[0] + sign_r * offset_steps, human[1] + sign_c * offset_steps),
        grid,
    )
    return target if is_floor(grid, target) else human


def move_monsters_controller(game: GameState, controller_name: str) -> List[Pos]:
    walls = walls_from_grid(game.grid)
    width = len(game.grid[0])
    height = len(game.grid)
    human_xy = (game.human[1], game.human[0])
    monsters_xy = many_row_col_to_xy(game.monsters)
    if controller_name == "greedy":
        ctrl = make_greedy_controller(
            walls,
            width=width,
            height=height,
            steps_per_turn=2,
            allow_stay=False,
            avoid_stacking=True,
        )
    else:
        ctrl = make_minimax_controller(
            walls,
            width=width,
            height=height,
            steps_per_turn=2,
            depth=2,
            player_can_stay=False,
            monster_can_stay=False,
        )
    paths = ctrl.decide(human_xy, monsters_xy)
    return many_xy_to_row_col(path[-1] for path in paths)


def move_human_with_bfs(game: GameState) -> Pos:
    human, _ = run_cpp_map_algorithm(
        "bfs",
        game.grid,
        game.human,
        game.monsters[0],
    )
    if not is_floor(game.grid, human):
        return game.human
    return human


@app.get("/")
def index():
    return INDEX_HTML


@app.post("/api/games")
def post_game():
    try:
        game = create_game(request.get_json(silent=True) or {})
    except (ValueError, CppAlgorithmError) as exc:
        return json_error(str(exc), 400)
    return jsonify(game.serialize()), 201


@app.get("/api/games/<game_id>")
def get_game(game_id: str):
    game = games.get(game_id)
    if game is None:
        return json_error("game not found", 404)
    return jsonify(game.serialize())


@app.post("/api/games/<game_id>/move")
def post_move(game_id: str):
    game = games.get(game_id)
    if game is None:
        return json_error("game not found", 404)
    try:
        apply_player_move(game, request.get_json(silent=True) or {})
    except (ValueError, CppAlgorithmError) as exc:
        return json_error(str(exc), 400)
    return jsonify(game.serialize())


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Maze Escape</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #18212f;
      --muted: #5c6675;
      --line: #c9d0da;
      --floor: #f7f3e8;
      --wall: #243241;
      --human: #2f9e44;
      --monster: #c92a2a;
      --panel: #ffffff;
      --accent: #2563eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #eef2f5;
      color: var(--ink);
    }
    main {
      display: grid;
      grid-template-columns: minmax(260px, 340px) 1fr;
      gap: 20px;
      min-height: 100vh;
      padding: 20px;
    }
    aside, section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    h1 { margin: 0 0 12px; font-size: 24px; }
    h2 { margin: 16px 0 8px; font-size: 16px; }
    label { display: block; margin: 8px 0 4px; color: var(--muted); font-size: 13px; }
    select, button {
      width: 100%;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--ink);
      font: inherit;
    }
    button {
      cursor: pointer;
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      font-weight: 650;
    }
    button.secondary {
      background: #f8fafc;
      color: var(--ink);
      border-color: var(--line);
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 12px;
    }
    .controls button:nth-child(1) { grid-column: 2; }
    .controls button:nth-child(2) { grid-column: 1; }
    .controls button:nth-child(3) { grid-column: 2; }
    .controls button:nth-child(4) { grid-column: 3; }
    .meta { color: var(--muted); font-size: 14px; line-height: 1.5; }
    .board-wrap { overflow: auto; }
    .board {
      display: grid;
      grid-template-columns: repeat(30, minmax(12px, 1fr));
      width: min(78vh, 100%);
      aspect-ratio: 1 / 1;
      border: 2px solid var(--wall);
      background: var(--wall);
    }
    .cell {
      position: relative;
      display: grid;
      place-items: center;
      aspect-ratio: 1 / 1;
      border: 1px solid rgba(24, 33, 47, 0.08);
      background: var(--floor);
      color: #172033;
      font-size: clamp(8px, 1.5vh, 14px);
      font-weight: 800;
      line-height: 1;
    }
    .wall { background: var(--wall); }
    .item-speed_boots { background: #ffd43b; }
    .item-home_stone { background: #4dabf7; color: #082f49; }
    .item-freeze_trap { background: #74c0fc; color: #0c4a6e; }
    .item-invisibility_cloak { background: #9775fa; color: #ffffff; }
    .trap {
      background: #e7f5ff;
      color: #075985;
      box-shadow: inset 0 0 0 2px #228be6;
    }
    .human { background: var(--human); }
    .monster { background: var(--monster); }
    .caught { background: #111827; }
    @media (max-width: 760px) {
      main { grid-template-columns: 1fr; padding: 12px; }
      .board { width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <aside>
      <h1>Maze Escape</h1>
      <h2>Escape mode</h2>
      <label for="escape-ai">Monster AI</label>
      <select id="escape-ai">
        <option value="astar">A*</option>
        <option value="greedy">Greedy</option>
        <option value="minimax">Minimax</option>
        <option value="sa">Simulated Annealing</option>
      </select>
      <label for="monster-count">A* monsters</label>
      <select id="monster-count">
        <option value="1">One</option>
        <option value="2">Two</option>
      </select>
      <button id="start-escape">Start escape</button>

      <h2>Chase mode</h2>
      <p class="meta">You control the monster one step at a time. Human uses BFS after every two monster steps.</p>
      <button id="start-chase">Start chase</button>

      <h2>Move</h2>
      <div class="controls">
        <button class="secondary" data-dir="up">Up</button>
        <button class="secondary" data-dir="left">Left</button>
        <button class="secondary" data-dir="down">Down</button>
        <button class="secondary" data-dir="right">Right</button>
      </div>
      <p id="meta" class="meta">No game running.</p>
    </aside>
    <section>
      <div id="board" class="board" aria-label="maze board"></div>
    </section>
  </main>
  <script>
    let game = null;
    const board = document.getElementById("board");
    const meta = document.getElementById("meta");

    async function startGame(mode) {
      const body = mode === "escape"
        ? {
            mode,
            opponent_ai: document.getElementById("escape-ai").value,
            monster_count: Number(document.getElementById("monster-count").value)
          }
        : { mode };
      meta.textContent = "Generating map...";
      const response = await fetch("/api/games", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      game = await response.json();
      render();
    }

    async function move(direction) {
      if (!game || game.status !== "running") return;
      const response = await fetch(`/api/games/${game.game_id}/move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ direction })
      });
      const payload = await response.json();
      if (!response.ok) {
        meta.textContent = payload.error || "Move failed";
        return;
      }
      game = payload;
      render();
    }

    function render() {
      board.innerHTML = "";
      if (!game) return;
      const humanKey = `${game.human.row},${game.human.col}`;
      const monsterKeys = new Set(game.monsters.map(m => `${m.row},${m.col}`));
      const itemsByKey = new Map(
        (game.items || []).map(item => [`${item.pos.row},${item.pos.col}`, item])
      );
      const trapsByKey = new Map(
        (game.traps || []).map(trap => [`${trap.pos.row},${trap.pos.col}`, trap])
      );
      for (let r = 0; r < game.grid.length; r++) {
        for (let c = 0; c < game.grid[r].length; c++) {
          const cell = document.createElement("div");
          const key = `${r},${c}`;
          const item = itemsByKey.get(key);
          const trap = trapsByKey.get(key);
          cell.className = "cell";
          if (game.grid[r][c] === 1) cell.classList.add("wall");
          if (item) {
            cell.classList.add(`item-${item.type}`);
            cell.textContent = item.symbol;
            cell.title = item.name;
          }
          if (trap) {
            cell.classList.add("trap");
            cell.textContent = trap.symbol;
            cell.title = `${trap.name} (${trap.lifetime} steps)`;
          }
          if (key === humanKey) cell.classList.add("human");
          if (monsterKeys.has(key)) cell.classList.add(key === humanKey ? "caught" : "monster");
          board.appendChild(cell);
        }
      }
      const pending = game.mode === "chase" && game.status === "running"
        ? ` | monster step: ${game.pending_monster_steps}/2`
        : "";
      const bonus = game.effects?.human_extra_steps > 0
        ? ` | speed steps: ${game.effects.human_extra_steps}`
        : "";
      meta.textContent = `${game.mode} | ${game.opponent_ai} | turns: ${game.step_count}${pending}${bonus} | ${game.status}${game.message ? " | " + game.message : ""}`;
    }

    document.getElementById("start-escape").addEventListener("click", () => startGame("escape"));
    document.getElementById("start-chase").addEventListener("click", () => startGame("chase"));
    document.querySelectorAll("[data-dir]").forEach(button => {
      button.addEventListener("click", () => move(button.dataset.dir));
    });
    document.addEventListener("keydown", event => {
      const map = { ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right" };
      if (map[event.key]) {
        event.preventDefault();
        move(map[event.key]);
      }
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)
