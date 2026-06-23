# Maze Escape: Human vs Monster AI Survival Game

Maze Escape is a small AI survival-chase project for the SOF106 Principles of
Artificial Intelligence group project. The project explores pathfinding,
search, and optimization algorithms on a 30 x 30 wrap-around maze, where a human
tries to escape and a monster tries to catch the human.

The repository currently contains the AI algorithm modules, a Flask backend,
a small browser-playable UI, generated map data, API tests, font assets, and a
LaTeX project report.

## Current Status

Implemented:

- Flask game backend and minimal playable web UI in `main.py`
- In-memory multi-game sessions addressed by `game_id`
- Escape mode, where the player controls the human and the monster is controlled
  by A*, Greedy, Minimax, or Simulated Annealing
- Chase mode, where the player controls the monster and the human is controlled
  by BFS escape
- Public AI package facade and C++ algorithm adapters in `ai/__init__.py`
- Genetic Algorithm map generation in `ai/genetic_map.cpp`
- A* monster pathfinding in `ai/astar.py`
- BFS-based human escape step selection in `ai/bfs_escape.cpp`
- Simulated Annealing monster movement in `ai/SA.cpp`
- Greedy multi-monster chase controller in `ai/greedy.py`
- Minimax multi-monster chase controller in `ai/minimax.py`
- Generated map file in `map/generated_map.txt`
- Flask API test-client coverage in `tests/test_flask_game.py`
- Project report source and PDF in `paper/`

Present but still empty / placeholder:

- `ui/hud.py`
- `ui/menu.py`
- `ui/renderer.py`

## Game Rules and Model

- The maze is represented as a 30 x 30 grid.
- `0` means a walkable floor cell.
- `1` means a wall.
- Movement uses four directions: up, down, left, and right.
- `stay` is not allowed by the Flask game API.
- The grid wraps around at the boundary. For example, moving left from column
  `0` enters column `29`.
- Public API coordinates are returned as objects: `{"row": 12, "col": 8}`.
- In escape mode, the human moves one step, then the selected monster AI runs
  once. A* single-monster mode advances up to two steps; A* two-monster mode
  moves one monster as a chaser and one as an interceptor; Greedy, Minimax, and
  Simulated Annealing use their controller output for the turn.
- In chase mode, the player-controlled monster moves one manual step per input
  and the UI updates immediately. After two monster inputs, the BFS-controlled
  human moves one step.
- The game has no fixed exit tile. The score is the number of interactions
  before the monster catches the human.

## Algorithms

### Genetic Algorithm Map Generation

`ai/genetic_map.cpp` generates random candidate maps and improves them over
multiple generations. The fitness function rewards:

- high connectivity between floor cells
- a wall ratio close to the target density
- useful junctions and branching paths
- reasonable average path distances

The Flask backend compiles the program on demand and runs it in a temporary
directory. It reads the generated 30 x 30 grid and spawn coordinates without
modifying the repository's `map/generated_map.txt`.

The human and monster spawn 30 to 40 BFS steps apart when a matching connected
pair is found.

### A* Monster Pathfinding

`ai/astar.py` implements A* search for the monster. It includes:

- `astar(grid, start, goal)`: returns a path from monster to human
- `get_next_step_single(grid, monster_pos, player_pos)`: returns the next move
  for one monster
- `get_next_steps_two(grid, m1_pos, m2_pos, player_pos)`: simple two-monster
  behavior where one monster chases and the other moves toward an intercept
  point

### BFS Human Escape

`ai/bfs_escape.cpp` reads `map/generated_map.txt`, runs BFS from the monster
position, and chooses the adjacent human move with the largest distance from the
monster. It writes the updated human position back to the map file.

The Flask backend uses this algorithm only in chase mode. It runs the executable
through a temporary map file so the repository's `map/generated_map.txt` is not
mutated during web play.

### Simulated Annealing Monster Movement

`ai/SA.cpp` reads `map/generated_map.txt`, builds an initial path from the
monster to the human, improves that path with Simulated Annealing, and writes the
monster's moved position back to the map file.

The SA objective still prioritizes short paths that move toward the human, but
now adds lightweight gameplay terms. Each SA run generates small random cell
costs for the early part of the path, rewards junction control, and can use the
previous SA move to discourage immediately backtracking. These terms act as
tie-breakers between similarly strong chase paths rather than replacing the
main pursuit objective.

The Flask backend exposes this as the `sa` monster AI in escape mode. Like BFS,
it is run through a temporary map file.

### Greedy Multi-Monster Chase

`ai/greedy.py` implements a Python monster controller for one or more monsters
on the wrap-around maze. It builds a `TorusGrid` representation with wall-aware
adjacency lists and cached BFS distance fields. Each monster then takes up to
`steps_per_turn` moves by choosing the neighboring cell with the smallest
distance to the current human position.

The controller can optionally allow monsters to stay in place and can avoid
stacking multiple monsters on the same cell unless that cell is the human's
position. It also provides helpers to load the generated map and choose extra
monster spawn points when the map file does not contain enough usable spawns.

Important functions and classes:

- `TorusGrid`: wrap-around grid model with walls, adjacency lists, and BFS
  distance caching
- `GreedyMonsterAI.decide(player_pos, monster_positions)`: returns each
  monster's planned path for the current turn
- `make_greedy_controller(...)`: creates a controller from explicit walls and
  grid dimensions
- `make_greedy_controller_from_map(...)`: creates a controller from
  `map/generated_map.txt`
- `pick_monster_spawns(...)`: selects reachable monster spawn cells away from
  the human

### Minimax Multi-Monster Chase

`ai/minimax.py` implements a lookahead monster controller on the same toroidal
grid model. It treats the chase as an alternating search: each monster moves for
its configured number of steps, then the human is allowed to choose a response.
Monsters minimize the human's safety score while the human maximizes it.

The search uses alpha-beta pruning, a small transposition table, move ordering,
and a distance-based evaluation function. Catching the human receives a very
large score swing, while non-terminal states are evaluated by the shortest
distances between the human and the monsters. The controller returns planned
per-monster paths for the current turn, so it can be used in the same style as
the greedy controller.

Important functions and classes:

- `TorusGrid`: wrap-around grid model shared in structure with the greedy
  controller
- `MinimaxMonsterAI.decide(player_pos, monster_positions)`: searches ahead and
  returns each monster's planned path for the current turn
- `make_minimax_controller(...)`: creates a controller from explicit walls and
  grid dimensions
- `make_minimax_controller_from_map(...)`: creates a controller from
  `map/generated_map.txt`
- `pick_monster_spawns(...)`: selects reachable monster spawn cells away from
  the human

## Repository Structure

```text
maze-escape-ai/
├── ai/
│   ├── __init__.py       # AI package facade and C++ adapter helpers
│   ├── astar.py          # A* pathfinding and two-monster movement helper
│   ├── bfs_escape.cpp    # BFS-based human escape move
│   ├── genetic_map.cpp   # Genetic Algorithm map generation
│   ├── SA.cpp            # Simulated Annealing monster movement
│   ├── greedy.py         # Greedy multi-monster chase controller
│   └── minimax.py        # Minimax multi-monster chase controller
├── assets/
│   └── fonts/            # bundled Munro font and license
├── map/
│   └── generated_map.txt # current generated map and coordinates
├── paper/
│   ├── main.tex          # report source
│   ├── main.pdf          # compiled report
│   └── sections/
├── tests/
│   └── test_flask_game.py
├── ui/                   # placeholder UI modules
├── main.py               # Flask backend, API routes, and minimal web UI
├── requirements.txt
└── README.md
```

## Map File Format

`map/generated_map.txt` is expected to contain:

1. 30 rows of grid data, with `0` and `1` separated by spaces
2. one line for the human position: `row column`
3. one line for the monster position: `row column`

Example tail:

```text
... 30 grid rows ...
12 8
20 17
```

The C++ files currently use the single-human / single-monster format. The Flask
backend preserves this by running C++ algorithms with temporary single-monster
map files.

## Setup

Flask 3.x supports Python 3.9 and newer. This project has been verified with a
Conda environment named `pai` using Python 3.12.

```bash
conda create -n pai python=3.12 -y
conda activate pai
python -m pip install -r requirements.txt
```

You can also use a virtual environment:

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Windows Command Prompt:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

### Windows C++ Compiler

Map generation, BFS, and Simulated Annealing use bundled C++17 programs. Install
one of these compilers and make sure it is available on `PATH`:

- MSYS2 / MinGW-w64 `g++`
- LLVM `clang++`
- Visual Studio Build Tools `cl` (run the project from a Developer PowerShell
  or Developer Command Prompt)

You can select a compiler explicitly with the `CXX` environment variable:

```powershell
$env:CXX = "g++"
python main.py
```

On Windows the backend builds `ai\bfs_escape.exe`, `ai\SA.exe`, and
`ai\genetic_map.exe` automatically. Existing macOS/Linux binaries are ignored.

## Running the Flask Game

Start the development server from the repository root:

```bash
conda activate pai
python main.py
```

Then open the local Flask URL printed by the server, usually:

```text
http://127.0.0.1:5000
```

The page has two entry points:

- Escape mode: choose a monster AI (`astar`, `greedy`, `minimax`, or `sa`) and
  move the human with arrow keys or the direction buttons.
- Chase mode: control the monster one direction at a time; the human responds
  with BFS after every two monster moves.

Each new game runs the bundled C++ Genetic Algorithm to generate a fresh
30 x 30 map and spawn coordinates.

Map generation prints GA progress in the server terminal and can take a moment.

## API Usage

Create a game:

```bash
curl -X POST http://127.0.0.1:5000/api/games \
  -H 'Content-Type: application/json' \
  -d '{"mode":"escape","opponent_ai":"astar","monster_count":1}'
```

Escape mode supports:

- `opponent_ai`: `astar`, `greedy`, `minimax`, or `sa`
- `monster_count`: `1` or `2` for `astar`; other AI modes force one monster

Chase mode:

```bash
curl -X POST http://127.0.0.1:5000/api/games \
  -H 'Content-Type: application/json' \
  -d '{"mode":"chase"}'
```

Fetch game state:

```bash
curl http://127.0.0.1:5000/api/games/<game_id>
```

Move in escape mode:

```bash
curl -X POST http://127.0.0.1:5000/api/games/<game_id>/move \
  -H 'Content-Type: application/json' \
  -d '{"direction":"up"}'
```

Swift Boots allows two separate one-tile move requests in one player turn, so
the second move may use a different direction. After the first move, the
response sets `waitingForSecondStep` to `true`. Submit another direction or end
the turn early:

```bash
curl -X POST http://127.0.0.1:5000/api/games/<game_id>/move \
  -H 'Content-Type: application/json' \
  -d '{"action":"end_turn"}'
```

The older `{ "endTurn": true }` form remains accepted as a compatibility
alias; Dash requests and controls have been removed.

Current durations are returned under `effects.speed_boots_turns`,
`effects.human_invisible_turns`, and `effects.monster_frozen_turns`.
The same response also exposes `speedBootsTurns`, `invisibleTurns`,
`remainingPlayerStepsThisTurn`, `maxPlayerStepsThisTurn`,
`waitingForSecondStep`, `gameOver`, `monster_states`, `pickedItems`, `items`,
`traps`, and a per-input `message` for frontend status rendering.

The Invisibility Cloak lasts for 10 completed player turns. While it is active,
monsters wander randomly instead of chasing and cannot capture the player. Its
duration decreases once when the complete player turn ends, not after each
movement step, so a two-step Swift Boots turn consumes only one cloak turn.

Frost Trap immediately freezes every monster for 5 completed player turns;
re-picking preserves the larger remaining duration rather than stacking it.
Teleport Stone ends the player movement phase and selects randomly from the top
10% safest valid cells, scored by distance to the nearest monster.

Move in chase mode:

```bash
curl -X POST http://127.0.0.1:5000/api/games/<game_id>/move \
  -H 'Content-Type: application/json' \
  -d '{"direction":"left"}'
```

In chase mode, each successful request immediately returns the monster's updated
position. The response includes `pending_monster_steps`; when it reaches the
second monster step, the backend runs BFS for the human and resets the pending
counter. Valid directions are `up`, `down`, `left`, and `right`. Invalid
directions, wall collisions, moves after the game ends, and unknown `game_id`
values return JSON errors.

## Running Tests

The Flask API tests use Python's built-in `unittest` module and monkeypatch the
slow map generation / C++ calls where needed.

```bash
conda activate pai
python -m py_compile ai/__init__.py main.py tests/test_flask_game.py
python -m unittest tests.test_flask_game
```

Expected result:

```text
Ran 8 tests

OK
```

## Running the C++ Algorithms

The C++ files read and write `../map/generated_map.txt` when run from the
`ai/` directory, so compile and execute them from there.

```bash
cd ai
g++ -std=c++17 bfs_escape.cpp -o bfs_escape
./bfs_escape
```

```bash
cd ai
g++ -std=c++17 SA.cpp -o SA
./SA
```

Both programs update `map/generated_map.txt` after computing the next move.

The Flask backend auto-detects these executables. If an executable is missing,
it tries the compiler configured by `CXX`, then `g++`, `clang++`, and MSVC
`cl`. If compilation fails, the related API request and browser UI show a clear
error while the Flask app remains usable.

## Report

The LaTeX report is stored in `paper/main.tex`, and the compiled PDF is stored
in `paper/main.pdf`. The report describes the broader project scope, including
Greedy Search, Minimax, item generation, UI, and algorithm comparison.

## Suggested Next Steps

- Split the inline HTML/CSS/JavaScript in `main.py` into templates and static
  assets when the UI grows.
- Add persistent score history or a leaderboard for survival / catch step
  counts.
- Add browser-level tests for the playable UI.
- Consider adding a faster map-generation preset for development.
- Keep the report and README aligned with the implemented code.
