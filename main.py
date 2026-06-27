from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import math
import random
import secrets
import threading
from typing import Any, Dict, List, Sequence, Tuple
import uuid


from flask import Flask, jsonify, request, send_from_directory

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
from models.human_predictor import (
    HumanDirectionPredictor,
    build_prediction_features,
    predicted_intercept_target,
)

Pos = Tuple[int, int]
ItemKind = str


MAP_GENERATION_ATTEMPTS = 4

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
    "speed_boots": "Swift Boots",
    "home_stone": "Safe Teleport Stone",
    "freeze_trap": "Frost Trap",
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
SPEED_BOOTS_DURATION = 5
SPEED_BOOTS_STEPS_PER_TURN = 2
TELEPORT_SAFE_DISTANCE = 6
FREEZE_TRAP_LIFETIME = 20
FREEZE_TRAP_DURATION = 5
INVISIBILITY_DURATION = 10

app = Flask(__name__)
try:
    human_predictor = HumanDirectionPredictor()
    print("Human direction predictor loaded.")
except Exception as exc:
    human_predictor = None
    print("Human direction predictor not loaded:", exc)

games: Dict[str, "GameState"] = {}
MAP_GENERATION_ATTEMPTS = 4
_generated_map_hashes: set[str] = set()
_map_hash_lock = threading.Lock()


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
    human_speed_boots_turns: int = 0
    human_invisible_turns: int = 0
    monster_frozen_turns: List[int] = field(default_factory=list)
    remaining_player_steps_this_turn: int = 0
    max_player_steps_this_turn: int = 1
    waiting_for_second_step: bool = False
    player_turn_new_effects: set[ItemKind] = field(default_factory=set)
    last_monster_paths: List[List[Pos]] = field(default_factory=list)
    sa_previous_move: Tuple[Pos, Pos] | None = None
    bfs_previous_human: Pos | None = None
    game_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    human_direction_history: List[str] = field(default_factory=list)
    turn_messages: List[str] = field(default_factory=list)
    turn_picked_items: List[ItemKind] = field(default_factory=list)
    map_seed: int | None = None

    def serialize(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "map_seed": self.map_seed,
            "grid": self.grid,
            "human": pos_to_json(self.human),
            "monsters": [
                {
                    **pos_to_json(pos),
                    "frozenTurns": (
                        self.monster_frozen_turns[index]
                        if index < len(self.monster_frozen_turns)
                        else 0
                    ),
                    "path": [
                        pos_to_json(path_pos)
                        for path_pos in (
                            self.last_monster_paths[index][1:]
                            if index < len(self.last_monster_paths)
                            else []
                        )
                    ],
                }
                for index, pos in enumerate(self.monsters)
            ],
            "monster_states": [
                {"pos": pos_to_json(pos), "frozen_turns": frozen}
                for pos, frozen in zip(self.monsters, self.monster_frozen_turns)
            ],
            "items": [item.serialize() for item in self.items if item.active],
            "traps": [trap.serialize() for trap in self.traps if trap.active],
            "effects": {
                "human_invisible_turns": self.human_invisible_turns,
                "speed_boots_turns": self.human_speed_boots_turns,
                "monster_frozen_turns": list(self.monster_frozen_turns),
            },
            "speedBootsTurns": self.human_speed_boots_turns,
            "invisibleTurns": self.human_invisible_turns,
            "isInvisible": self.human_invisible_turns > 0,
            "invisible": self.human_invisible_turns > 0,
            "remainingPlayerStepsThisTurn": self.remaining_player_steps_this_turn,
            "maxPlayerStepsThisTurn": self.max_player_steps_this_turn,
            "waitingForSecondStep": self.waiting_for_second_step,
            "canEndTurn": (
                self.mode == "escape"
                and self.status == "running"
                and self.waiting_for_second_step
            ),
            "gameOver": self.status == "ended",
            "outcome": "loss" if self.status == "ended" else None,
            "won": False,
            "lost": self.status == "ended",
            "pickedItems": list(self.turn_picked_items),
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
    return jsonify({"error": message, "message": message}), status_code


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
        raise ValueError("Blocked by wall.")
    return nxt


def walkable_neighbors(grid: Sequence[Sequence[int]], pos: Pos) -> List[Pos]:
    neighbors: List[Pos] = []
    for dr, dc in VALID_DIRECTIONS.values():
        nxt = wrap_pos((pos[0] + dr, pos[1] + dc), grid)
        if is_floor(grid, nxt):
            neighbors.append(nxt)
    return neighbors


def assert_running(game: GameState) -> None:
    if game.status != "running":
        raise ValueError("game has already ended")


def update_status(game: GameState) -> None:
    if game.human_invisible_turns <= 0 and any(
        monster == game.human for monster in game.monsters
    ):
        game.status = "ended"
        game.message = f"Monster caught the player after {game.step_count} turns."


def parse_spawn_line(lines: Sequence[str], label: str) -> Pos | None:
    """Parse lines such as 'Human spawn: 17 18'."""
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith(label):
            continue

        parts = stripped.replace(":", " ").split()
        numbers: list[int] = []
        for part in parts:
            if part.lstrip("-").isdigit():
                numbers.append(int(part))

        if len(numbers) >= 2:
            return numbers[0], numbers[1]

    return None


def parse_genetic_map_from_debug_output(debug_text: str) -> tuple[List[List[int]], Pos, Pos]:
    """
    Fallback parser for genetic_map.exe debug output.

    Some versions of genetic_map.exe print a visual 30x30 map to stdout/stderr:
        . = floor
        # = wall
        H = human spawn
        M = monster spawn

    If run_cpp_genetic_map() treats that output as a failure, this parser
    extracts grid, human spawn, and monster spawn directly from the debug text.
    """
    lines = debug_text.splitlines()

    map_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) >= 30 and all(ch in ".#HM" for ch in stripped[:30]):
            map_lines.append(stripped[:30])

    if len(map_lines) < 30:
        raise ValueError("could not parse generated map from genetic_map.exe output")

    map_lines = map_lines[:30]

    grid: List[List[int]] = []
    human: Pos | None = None
    monster: Pos | None = None

    for row_index, line in enumerate(map_lines):
        row: list[int] = []
        for col_index, char in enumerate(line):
            if char == "#":
                row.append(1)
            else:
                row.append(0)

            if char == "H":
                human = (row_index, col_index)
            elif char == "M":
                monster = (row_index, col_index)

        grid.append(row)

    # If H/M are not embedded in the visual map, try the explicit spawn lines.
    if human is None:
        human = parse_spawn_line(lines, "Human spawn")

    if monster is None:
        monster = parse_spawn_line(lines, "Monster spawn")

    if human is None or monster is None:
        raise ValueError("could not parse human or monster spawn from genetic_map.exe output")

    return grid, human, monster


def load_genetic_map(seed: int) -> tuple[List[List[int]], Pos, Pos]:
    """
    Load the map from the C++ genetic generator.

    Normal path:
        use ai.run_cpp_genetic_map()

    Compatibility path:
        if run_cpp_genetic_map() raises CppAlgorithmError but the C++ program
        printed a complete visual map, parse that output instead.
    """
    try:
        return run_cpp_genetic_map(seed=seed)
    except CppAlgorithmError as exc:
        return parse_genetic_map_from_debug_output(str(exc))


def map_layout_hash(grid: Sequence[Sequence[int]]) -> str:
    """Return a stable hash for duplicate-layout detection."""
    encoded = "\n".join("".join(str(int(cell)) for cell in row) for row in grid)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def generate_unique_game_map() -> tuple[List[List[int]], Pos, Pos, int]:
    """Generate a map not currently used by another in-memory game."""
    for _ in range(MAP_GENERATION_ATTEMPTS):
        seed = secrets.randbits(64)
        grid, human, monster = load_genetic_map(seed)
        layout_hash = map_layout_hash(grid)
        with _map_hash_lock:
            if layout_hash in _generated_map_hashes:
                continue
            _generated_map_hashes.add(layout_hash)
            return grid, human, monster, seed
    raise ValueError(
        f"map generator returned a duplicate layout {MAP_GENERATION_ATTEMPTS} times"
    )


def create_game(payload: dict[str, Any]) -> GameState:
    mode = str(payload.get("mode", "escape")).lower()
    print("Create game payload:", payload)
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

    grid, human, first_monster, map_seed = generate_unique_game_map()
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
        map_seed=map_seed,
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
    del min_distance  # Kept for compatibility; selection is always from the safest 10%.
    distances = distance_field(game.grid, game.monsters)
    occupied = set(game.monsters)
    occupied.update(item.pos for item in game.items if item.active)
    occupied.update(trap.pos for trap in game.traps if trap.active)
    candidates = [
        (distances[row][col], (row, col))
        for row, cells in enumerate(game.grid)
        for col, value in enumerate(cells)
        if int(value) == 0
        and distances[row][col] is not None
        and (row, col) != game.human
        and (row, col) not in occupied
    ]
    if not candidates:
        return game.human
    candidates.sort(key=lambda entry: entry[0], reverse=True)
    safest_count = max(1, math.ceil(len(candidates) * 0.10))
    return random.choice([pos for _, pos in candidates[:safest_count]])


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


def decay_effects_after_player_turn(
    game: GameState, newly_applied: set[ItemKind] | None = None
) -> None:
    newly_applied = newly_applied or set()
    update_trap_lifetimes(game)
    if "speed_boots" not in newly_applied and game.human_speed_boots_turns > 0:
        game.human_speed_boots_turns -= 1
    if "invisibility_cloak" not in newly_applied and game.human_invisible_turns > 0:
        game.human_invisible_turns -= 1
    if "freeze_trap" not in newly_applied:
        game.monster_frozen_turns = [
            max(0, frozen - 1) for frozen in game.monster_frozen_turns
        ]


def apply_item_effect(game: GameState, item: ItemState) -> bool:
    """Apply a picked-up item. Return True when movement should stop."""

    game.turn_picked_items.append(item.type)
    if item.type == "speed_boots":
        game.human_speed_boots_turns = SPEED_BOOTS_DURATION
        game.turn_messages.append(
            "Swift Boots activated: move twice per turn for 5 turns."
        )
        item.active = False
        return False
    if item.type == "home_stone":
        game.human = find_safe_teleport_position(game)
        game.turn_messages.append(
            "Teleport Stone activated: moved to a safe tile."
        )
        item.active = False
        return True
    if item.type == "freeze_trap":
        while len(game.monster_frozen_turns) < len(game.monsters):
            game.monster_frozen_turns.append(0)
        for index in range(len(game.monsters)):
            game.monster_frozen_turns[index] = max(
                game.monster_frozen_turns[index], FREEZE_TRAP_DURATION
            )
        game.turn_messages.append("Frost Trap activated: all monsters frozen.")
        item.active = False
        return False
    if item.type == "invisibility_cloak":
        game.human_invisible_turns = max(
            game.human_invisible_turns, INVISIBILITY_DURATION
        )
        game.turn_messages.append("Cloak activated: invisible for 10 turns.")
        game.cloak_already_spawned = True
        item.active = False
        return False
    return False


def check_player_item_pickup(game: GameState) -> tuple[bool, ItemKind | None]:
    for item in game.items:
        if item.active and item.pos == game.human:
            return apply_item_effect(game, item), item.type
    return False, None


def begin_escape_player_turn(game: GameState) -> None:
    game.max_player_steps_this_turn = (
        SPEED_BOOTS_STEPS_PER_TURN
        if game.human_speed_boots_turns > 0
        else 1
    )
    game.remaining_player_steps_this_turn = game.max_player_steps_this_turn
    game.waiting_for_second_step = False
    game.player_turn_new_effects.clear()


def reset_escape_player_turn(game: GameState) -> None:
    game.remaining_player_steps_this_turn = 0
    game.waiting_for_second_step = False
    game.player_turn_new_effects.clear()


def move_human_with_items(game: GameState, direction: str) -> tuple[bool, bool]:
    """Move one tile. Return (moved, teleport_triggered)."""
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"invalid direction: {direction}")
    dr, dc = VALID_DIRECTIONS[direction]
    nxt = wrap_pos((game.human[0] + dr, game.human[1] + dc), game.grid)
    if not is_floor(game.grid, nxt):
        return False, False

    game.human = nxt
    teleport_triggered, picked_type = check_player_item_pickup(game)
    if picked_type:
        game.player_turn_new_effects.add(picked_type)
    game.human_direction_history.append(direction)
    if len(game.human_direction_history) > 10:
        game.human_direction_history = game.human_direction_history[-10:]
    return True, teleport_triggered


def finish_escape_player_turn(game: GameState) -> None:
    decay_effects_after_player_turn(game, game.player_turn_new_effects)
    game.remaining_player_steps_this_turn = 0
    game.waiting_for_second_step = False

    previous_monsters = list(game.monsters)
    if not catches_human(game):
        game.monsters = move_monsters_with_item_effects(game)
    if game.monsters != previous_monsters:
        game.turn_messages.append("Turn ended. Monsters moved.")

    game.step_count += 1
    if game.step_count % ITEM_SPAWN_INTERVAL == 0:
        spawn_random_items(game, ITEM_SPAWN_COUNT)
    update_status(game)
    if game.status == "running":
        game.message = "; ".join(game.turn_messages)
    game.player_turn_new_effects.clear()


def apply_player_move(game: GameState, payload: dict[str, Any]) -> None:
    assert_running(game)
    game.turn_messages.clear()
    game.turn_picked_items.clear()
    game.message = ""
    if game.mode == "escape":
        if (
            payload.get("action") == "end_turn"
            or payload.get("endTurn") is True
        ):
            if not game.waiting_for_second_step:
                raise ValueError("there is no unfinished player turn")
            game.turn_messages.append("Player ended the turn early")
            finish_escape_player_turn(game)
            return

        direction = str(payload.get("direction", "")).lower()
        if direction not in VALID_DIRECTIONS:
            raise ValueError(f"invalid direction: {direction}")
        if game.remaining_player_steps_this_turn <= 0:
            begin_escape_player_turn(game)

        moved, teleport_triggered = move_human_with_items(game, direction)
        if not moved:
            if game.waiting_for_second_step:
                game.turn_messages.append("Second move blocked; turn ended")
                finish_escape_player_turn(game)
                return
            reset_escape_player_turn(game)
            raise ValueError("Blocked by wall.")

        game.remaining_player_steps_this_turn -= 1
        if catches_human(game):
            finish_escape_player_turn(game)
            return
        if teleport_triggered:
            game.turn_messages.append("Teleport ended the movement phase")
            finish_escape_player_turn(game)
            return
        if game.remaining_player_steps_this_turn > 0:
            game.waiting_for_second_step = True
            game.message = "; ".join(
                game.turn_messages
                + ["Swift Boots active: choose your second move or end turn."]
            )
            return

        finish_escape_player_turn(game)
        return
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
    update_status(game)


def catches_human(game: GameState) -> bool:
    if game.human_invisible_turns > 0:
        return False
    return any(monster == game.human for monster in game.monsters)


def check_monster_trap(game: GameState, monster_index: int) -> bool:
    for trap in game.traps:
        if trap.active and trap.pos == game.monsters[monster_index]:
            game.monster_frozen_turns[monster_index] = max(
                game.monster_frozen_turns[monster_index], FREEZE_TRAP_DURATION
            )
            trap.active = False
            return True
    return False


def remember_sa_move(game: GameState, start: Pos, end: Pos) -> None:
    if game.opponent_ai == "sa" and start != end:
        game.sa_previous_move = (start, end)


def move_monsters_with_item_effects(game: GameState) -> List[Pos]:
    while len(game.monster_frozen_turns) < len(game.monsters):
        game.monster_frozen_turns.append(0)

    if game.human_invisible_turns > 0:
        next_monsters = list(game.monsters)
        actual_paths: List[List[Pos]] = [[pos] for pos in game.monsters]
        for index, start in enumerate(game.monsters):
            if game.monster_frozen_turns[index] > 0:
                continue
            for _ in range(2):
                neighbors = walkable_neighbors(game.grid, next_monsters[index])
                alternatives = [pos for pos in neighbors if pos != game.human]
                choices = alternatives or neighbors
                if not choices:
                    break
                next_monsters[index] = random.choice(choices)
                game.monsters[index] = next_monsters[index]
                actual_paths[index].append(next_monsters[index])
                if check_monster_trap(game, index):
                    break
        game.last_monster_paths = actual_paths
        return next_monsters

    if all(
        frozen > 0
        for frozen in game.monster_frozen_turns[:len(game.monsters)]
    ):
        game.last_monster_paths = [[pos] for pos in game.monsters]
        return list(game.monsters)

    previous_monsters = list(game.monsters)
    paths = planned_monster_paths(game)
    next_monsters = list(game.monsters)
    actual_paths: List[List[Pos]] = [[pos] for pos in game.monsters]
    for index, path in enumerate(paths):
        if index >= len(next_monsters):
            break
        if game.monster_frozen_turns[index] > 0:
            continue

        for pos in path[1:]:
            next_monsters[index] = pos
            game.monsters[index] = pos
            actual_paths[index].append(pos)
            if check_monster_trap(game, index):
                break
            if pos == game.human:
                break
    game.last_monster_paths = actual_paths
    if next_monsters:
        remember_sa_move(game, previous_monsters[0], next_monsters[0])
    return next_monsters


def planned_monster_paths(game: GameState) -> List[List[Pos]]:
    ai_name = game.opponent_ai
    if ai_name == "astar":
        return planned_astar_paths(game)
    if ai_name in {"greedy", "minimax"}:
        return planned_controller_paths(game, ai_name)
    if ai_name == "sa":
        return [planned_sa_path(game)]
    raise ValueError(f"unsupported monster AI: {ai_name}")


def planned_sa_path(game: GameState) -> List[Pos]:
    start = game.monsters[0]
    fallback = limited_path(game.grid, start, game.human, 2)
    try:
        _, monster = run_cpp_map_algorithm(
            "sa",
            game.grid,
            game.human,
            start,
            sa_previous_move=game.sa_previous_move,
        )
    except CppAlgorithmError:
        return fallback

    if not valid_sa_destination(game.grid, start, monster, game.human):
        return fallback
    return [start, monster]


def valid_sa_destination(
    grid: Sequence[Sequence[int]],
    start: Pos,
    destination: Pos,
    human: Pos,
) -> bool:
    if not is_floor(grid, destination):
        return False
    path = astar([list(row) for row in grid], start, destination)
    if not path or len(path) - 1 > 2:
        return False
    can_advance_toward_human = len(limited_path(grid, start, human, 1)) > 1
    if destination == start and start != human and can_advance_toward_human:
        return False
    return True


def planned_astar_paths(game: GameState) -> List[List[Pos]]:
    if len(game.monsters) == 1:
        return [limited_path(game.grid, game.monsters[0], game.human, 2)]

    m1, m2 = game.monsters[:2]
    d1 = torus_manhattan(m1, game.human, game.grid)
    d2 = torus_manhattan(m2, game.human, game.grid)

    if d1 <= d2:
        chaser = m1
        interceptor = m2

        chaser_path = limited_path(
            game.grid,
            chaser,
            game.human,
            2,
        )

        intercept_target = get_prediction_intercept_target(
            game,
            reference_monster=chaser,
            fallback_monster=interceptor,
        )

        interceptor_path = limited_path(
            game.grid,
            interceptor,
            intercept_target,
            1,
        )

        return [chaser_path, interceptor_path]

    chaser = m2
    interceptor = m1

    interceptor_target = get_prediction_intercept_target(
        game,
        reference_monster=chaser,
        fallback_monster=interceptor,
    )

    return [
        limited_path(
            game.grid,
            interceptor,
            interceptor_target,
            1,
        ),
        limited_path(
            game.grid,
            chaser,
            game.human,
            2,
        ),
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
        path = planned_sa_path(game)
        monster = path[-1]
        remember_sa_move(game, game.monsters[0], monster)
        return [monster]
    raise ValueError(f"unsupported monster AI: {ai_name}")


def move_monsters_astar(game: GameState) -> List[Pos]:
    if len(game.monsters) == 1:
        return [
            advance_along_path(
                game.grid,
                game.monsters[0],
                game.human,
                2,
            )
        ]

    astar_monster = game.monsters[0]
    prediction_monster = game.monsters[1]

    # Monster 1: direct A* chasing
    new_astar_monster = advance_along_path(
        game.grid,
        astar_monster,
        game.human,
        2,
    )

    # Monster 2: prediction-guided interception
    intercept_target = get_prediction_intercept_target(game)

    new_prediction_monster = advance_along_path(
        game.grid,
        prediction_monster,
        intercept_target,
        2,
    )

    return [new_astar_monster, new_prediction_monster]

def get_prediction_intercept_target(
    game: GameState,
    reference_monster: Pos,
    fallback_monster: Pos,
) -> Pos:
    """
    Use the Random Forest model to predict the player's next direction.
    Then convert the predicted direction into an interception target.

    If prediction fails, fall back to the original intercept_point logic.
    """

    if human_predictor is None:
        return intercept_point(
            game.grid,
            fallback_monster,
            game.human,
        )

    try:
        feature = build_prediction_features(
            grid=game.grid,
            human=game.human,
            reference_monster=reference_monster,
            history=game.human_direction_history,
        )

        predicted_direction = human_predictor.predict_direction(feature)

        target = predicted_intercept_target(
            grid=game.grid,
            human=game.human,
            predicted_direction=predicted_direction,
            distance=3,
        )

        print(
            "[Prediction]",
            "direction:",
            predicted_direction,
            "target:",
            target,
        )

        return target

    except Exception as exc:
        print("[Prediction failed] fallback to intercept point:", exc)

        return intercept_point(
            game.grid,
            fallback_monster,
            game.human,
        )

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
    previous_human = game.human
    human, _ = run_cpp_map_algorithm(
        "bfs",
        game.grid,
        game.human,
        game.monsters[0],
        bfs_previous_human=game.bfs_previous_human,
    )
    if not is_floor(game.grid, human):
        return game.human
    game.bfs_previous_human = previous_human
    return human


@app.get("/")
def index():
    return send_from_directory("ui/index", "index.html")


@app.route("/ui/<path:filename>")
def ui_static(filename):
    return send_from_directory("ui", filename)


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




if __name__ == "__main__":
    app.run(debug=True)
