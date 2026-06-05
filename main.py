from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple
import uuid

from flask import Flask, jsonify, request

from ai import (
    CppAlgorithmError,
    astar,
    generate_map_ga,
    get_approx_torus_spawn_points,
    make_greedy_controller,
    make_minimax_controller,
    many_row_col_to_xy,
    many_xy_to_row_col,
    run_cpp_map_algorithm,
    walls_from_grid,
)

Pos = Tuple[int, int]

VALID_DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ESCAPE_AIS = {"astar", "greedy", "minimax", "sa"}
CHASE_AI = "bfs"

app = Flask(__name__)
games: Dict[str, "GameState"] = {}


@dataclass
class GameState:
    grid: List[List[int]]
    human: Pos
    monsters: List[Pos]
    mode: str
    opponent_ai: str
    step_count: int = 0
    pending_monster_steps: int = 0
    status: str = "running"
    message: str = ""
    game_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def serialize(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "grid": self.grid,
            "human": pos_to_json(self.human),
            "monsters": [pos_to_json(pos) for pos in self.monsters],
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

    grid = generate_map_ga(
        pop_size=30,
        generations=20,
        mutation_rate=0.01,
        elite_num=5,
    )
    if grid is None:
        raise ValueError("map generation failed")

    human, first_monster = get_approx_torus_spawn_points(grid)
    monsters = [first_monster]
    while len(monsters) < monster_count:
        extra = find_extra_monster_spawn(grid, human, monsters)
        if extra is None:
            raise ValueError("could not place the second monster")
        monsters.append(extra)

    game = GameState(
        grid=[list(row) for row in grid],
        human=human,
        monsters=monsters,
        mode=mode,
        opponent_ai=opponent_ai,
    )
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


def apply_player_move(game: GameState, payload: dict[str, Any]) -> None:
    assert_running(game)
    if game.mode == "escape":
        direction = str(payload.get("direction", "")).lower()
        game.human = move_one(game.grid, game.human, direction)
        if not catches_human(game):
            game.monsters = move_monsters(game)
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
    return any(monster == game.human for monster in game.monsters)


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
      aspect-ratio: 1 / 1;
      border: 1px solid rgba(24, 33, 47, 0.08);
      background: var(--floor);
    }
    .wall { background: var(--wall); }
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
      for (let r = 0; r < game.grid.length; r++) {
        for (let c = 0; c < game.grid[r].length; c++) {
          const cell = document.createElement("div");
          const key = `${r},${c}`;
          cell.className = "cell";
          if (game.grid[r][c] === 1) cell.classList.add("wall");
          if (key === humanKey) cell.classList.add("human");
          if (monsterKeys.has(key)) cell.classList.add(key === humanKey ? "caught" : "monster");
          board.appendChild(cell);
        }
      }
      const pending = game.mode === "chase" && game.status === "running"
        ? ` | monster step: ${game.pending_monster_steps}/2`
        : "";
      meta.textContent = `${game.mode} | ${game.opponent_ai} | turns: ${game.step_count}${pending} | ${game.status}${game.message ? " | " + game.message : ""}`;
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
