"""Headless test: pit each algorithm against a smart fleeing player."""
import random, time
from monster_ai import TorusGrid, make_monster_controller

W = H = 30
STEPS = 2


def random_walls(seed, density=0.12):
    rng = random.Random(seed)
    return {(rng.randrange(W), rng.randrange(H))
            for _ in range(int(W * H * density))}


def smart_flee(grid, player, monsters):
    """Player moves 1 cell to maximise distance to the nearest monster
    (true shortest-path distance) -- a tough, evasive opponent."""
    fields = [grid.distance_field(grid.coord(grid.index(m))) for m in monsters]
    p = grid.index(player)
    best, best_score = p, -1
    cands = list(grid.adj[p]) + [p]  # may stand still
    for j in cands:
        score = min(f[j] if f[j] >= 0 else 0 for f in fields)
        if score > best_score:
            best_score, best = score, j
    return grid.coord(best)


def run(algorithm, walls, seed, max_turns=400, depth=2):
    grid = TorusGrid(W, H, walls)
    free = [grid.coord(i) for i in range(grid.n) if not grid.blocked[i]]
    rng = random.Random(seed)
    player = rng.choice(free)
    monsters = []
    while len(monsters) < 2:
        c = rng.choice(free)
        if c != player and c not in monsters and grid.toroidal_manhattan(c, player) > 6:
            monsters.append(c)

    ctrl = make_monster_controller(algorithm, walls, W, H, STEPS, depth=depth)
    total_t, calls = 0.0, 0
    for turn in range(1, max_turns + 1):
        # 1) player flees one step
        player = smart_flee(grid, player, monsters)
        if player in monsters:
            return turn, total_t / max(calls, 1)
        # 2) monsters think + move
        t0 = time.perf_counter()
        paths = ctrl.decide(player, monsters)
        total_t += time.perf_counter() - t0
        calls += 1
        monsters = [pth[-1] for pth in paths]
        # capture if a monster passed through the player at any step
        caught = any(player in pth for pth in paths)
        if caught:
            return turn, total_t / calls
    return None, total_t / max(calls, 1)


def main():
    print(f"{'algo':8} {'seed':4} {'turns_to_catch':>14} {'avg_ms/turn':>12}")
    for seed in range(4):
        walls = random_walls(seed)
        for algo in ("greedy", "minimax"):
            turns, avg = run(algo, walls, seed, depth=2)
            label = turns if turns is not None else "ESCAPED"
            print(f"{algo:8} {seed:<4} {str(label):>14} {avg*1000:>12.3f}")
    # deeper minimax timing on an open map
    print("\nminimax depth scaling (open map, single decision):")
    grid = TorusGrid(W, H, set())
    ctrl_meta = None
    player = (15, 15)
    monsters = [(2, 2), (27, 27)]
    for depth in (1, 2, 3, 4):
        ctrl = make_monster_controller("minimax", set(), W, H, STEPS, depth=depth)
        t0 = time.perf_counter()
        ctrl.decide(player, monsters)
        print(f"  depth={depth} rounds  ->  {(time.perf_counter()-t0)*1000:8.2f} ms")


if __name__ == "__main__":
    main()
