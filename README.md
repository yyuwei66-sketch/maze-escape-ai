# Maze Escape: Human vs Monster AI Survival Game

Maze Escape is a small AI survival-chase project for the SOF106 Principles of
Artificial Intelligence group project. The project explores pathfinding,
search, and optimization algorithms on a 30 x 30 wrap-around maze, where a human
tries to escape and a monster tries to catch the human.

The repository currently contains the AI algorithm modules, generated map data,
basic test code, font assets, and a LaTeX project report. A complete playable
game loop / UI entry point is not present yet, so this README describes the
implemented files as they exist in the repository today.

## Current Status

Implemented:

- Genetic Algorithm map generation in `ai/genetic_map.py`
- A* monster pathfinding in `ai/astar.py`
- BFS-based human escape step selection in `ai/bfs_escape.cpp`
- Simulated Annealing monster movement in `ai/SA.cpp`
- Generated map file in `map/generated_map.txt`
- Basic Genetic Algorithm smoke test in `tests/test_genetic_map.py`
- Project report source and PDF in `paper/`

Present but still empty / placeholder:

- `ai/greedy.py`
- `ai/minimax.py`
- `ui/hud.py`
- `ui/menu.py`
- `ui/renderer.py`

## Game Rules and Model

- The maze is represented as a 30 x 30 grid.
- `0` means a walkable floor cell.
- `1` means a wall.
- Movement uses four directions: up, down, left, and right.
- The grid wraps around at the boundary. For example, moving left from column
  `0` enters column `29`.
- The human moves one step per turn.
- The monster movement modules are designed around the monster moving up to two
  steps per turn.

## Algorithms

### Genetic Algorithm Map Generation

`ai/genetic_map.py` generates random candidate maps and improves them over
multiple generations. The fitness function rewards:

- high connectivity between floor cells
- a wall ratio close to the target density
- useful junctions and branching paths
- reasonable average path distances

Important functions:

- `generate_map_ga(...)`: generates and returns the best 30 x 30 map
- `fitness(grid)`: scores a map
- `get_valid_spawn_points(grid, min_dist=10, max_dist=25)`: selects human and
  monster spawn points on valid floor cells
- `bfs_distance(start, end, grid)`: shortest path distance on the wrap-around map

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

### Simulated Annealing Monster Movement

`ai/SA.cpp` reads `map/generated_map.txt`, builds an initial path from the
monster to the human, improves that path with Simulated Annealing, and writes the
monster's moved position back to the map file.

## Repository Structure

```text
maze-escape-ai/
├── ai/
│   ├── astar.py          # A* pathfinding and two-monster movement helper
│   ├── bfs_escape.cpp    # BFS-based human escape move
│   ├── genetic_map.py    # Genetic Algorithm map generation
│   ├── SA.cpp            # Simulated Annealing monster movement
│   ├── greedy.py         # placeholder
│   └── minimax.py        # placeholder
├── assets/
│   └── fonts/            # bundled Munro font and license
├── map/
│   └── generated_map.txt # current generated map and coordinates
├── paper/
│   ├── main.tex          # report source
│   ├── main.pdf          # compiled report
│   └── sections/
├── tests/
│   └── test_genetic_map.py
├── ui/                   # placeholder UI modules
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

`ai/astar.py` can also parse a two-monster format when three coordinate lines
are present near the end of a map file, but the C++ files currently use the
single-human / single-monster format.

## Setup

The repository has no required third-party Python packages at the moment;
`requirements.txt` is currently empty. Python 3.9+ is recommended because the
Python modules use modern type hints.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running the Python Smoke Test

The current test file is a script-style smoke test rather than a pytest test
case. It generates a small Genetic Algorithm map and prints spawn information.

```bash
python tests/test_genetic_map.py
```

Expected output includes the generated map size, selected human and monster
positions, and their BFS distance.

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

## Report

The LaTeX report is stored in `paper/main.tex`, and the compiled PDF is stored
in `paper/main.pdf`. The report describes the intended full project scope,
including Greedy Search, Minimax, item generation, UI, and algorithm comparison.
Some of those parts are still placeholders in the current codebase.

## Suggested Next Steps

- Add a main game loop that connects map generation, human movement, monster
  movement, collision checks, and turn progression.
- Implement the placeholder Greedy and Minimax Python modules.
- Fill in the UI modules or remove them until the UI is ready.
- Convert `tests/test_genetic_map.py` into proper pytest assertions.
- Add a script that generates `map/generated_map.txt` from
  `ai/genetic_map.py`.
- Keep the report and README aligned with the implemented code.
