"""
evaluate.py  ——  Monster AI Algorithm Evaluation

Maps:    genetic_map.py real genetic algorithm (not simulated fake data)
Player:  BFS escape (maximise distance from nearest monster each step)
         — identical across ALL algorithms, only variable is the monster AI
Algorithms: A* / Greedy / Minimax / SA  
Output:  6 independent PNG charts

Usage:
  python evaluate.py --games 30
  python evaluate.py --games 30 --skip sa     # skip SA if C++ not compiled
  python evaluate.py --fast                   # lightweight GA, quick test
"""
from __future__ import annotations

import argparse, os, sys, time, random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

try:
    from genetic_map import generate_map_ga, get_approx_torus_spawn_points
except ImportError:
    from ai.genetic_map import generate_map_ga, get_approx_torus_spawn_points

# ── 严格匹配 ai/__init__.py 导出的接口 ──
try:
    from ai import (
        astar, 
        make_greedy_controller, 
        make_minimax_controller,
        many_row_col_to_xy, 
        many_xy_to_row_col,
        run_cpp_map_algorithm, 
        walls_from_grid,
    )
    HAS_AI = True
except ImportError as e:
    HAS_AI = False
    print(f"[WARN] ai module 导入失败 ({e}) — Greedy/Minimax/SA 将回退到 BFS A*")

Pos = Tuple[int, int]

VALID_DIRECTIONS = {
    "up":(-1,0), "down":(1,0), "left":(0,-1), "right":(0,1),
}
MAX_STEPS = 300
GA_FULL   = dict(pop_size=30, generations=20, mutation_rate=0.01, elite_num=5)
GA_FAST   = dict(pop_size=15, generations=8,  mutation_rate=0.01, elite_num=3)
_ga_params = GA_FULL

# ════════════════════════════════════════════════════════════════════════════
# Real map generation
# ════════════════════════════════════════════════════════════════════════════

def get_map() -> Tuple[List[List[int]], Pos, Pos]:
    grid = generate_map_ga(**_ga_params)
    human, monster = get_approx_torus_spawn_points(grid)
    return [list(row) for row in grid], human, monster

# ════════════════════════════════════════════════════════════════════════════
# Grid utilities
# ════════════════════════════════════════════════════════════════════════════

def wrap(pos: Pos, grid) -> Pos:
    return pos[0] % len(grid), pos[1] % len(grid[0])

def passable(grid, pos: Pos) -> bool:
    r, c = wrap(pos, grid)
    return int(grid[r][c]) == 0

def torus_dist(a: Pos, b: Pos, grid) -> int:
    h, w = len(grid), len(grid[0])
    dr = abs(a[0]-b[0]); dr = min(dr, h-dr)
    dc = abs(a[1]-b[1]); dc = min(dc, w-dc)
    return dr + dc

# ════════════════════════════════════════════════════════════════════════════
# Player: BFS escape (same for all algorithms)
# ════════════════════════════════════════════════════════════════════════════

def player_move(grid, human: Pos, monsters: List[Pos]) -> Optional[str]:
    nearest = min(monsters, key=lambda m: torus_dist(human, m, grid))
    H, W = len(grid), len(grid[0])
    best_dir, best_d = None, -1
    for d, (dr, dc) in VALID_DIRECTIONS.items():
        nxt = ((human[0]+dr)%H, (human[1]+dc)%W)
        if passable(grid, nxt):
            dv = torus_dist(nxt, nearest, grid)
            if dv > best_d:
                best_d, best_dir = dv, d
    return best_dir

# ════════════════════════════════════════════════════════════════════════════
# Monster AIs
# ════════════════════════════════════════════════════════════════════════════

def _bfs(grid, start: Pos, goal: Pos) -> List[Pos]:
    H, W = len(grid), len(grid[0])
    q = deque([(start, [start])]); visited = {start}
    while q:
        pos, path = q.popleft()
        if pos == goal: return path
        for dr, dc in VALID_DIRECTIONS.values():
            nxt = ((pos[0]+dr)%H, (pos[1]+dc)%W)
            if nxt not in visited and passable(grid, nxt):
                visited.add(nxt); q.append((nxt, path+[nxt]))
    return [start]

def _path(grid, s: Pos, g: Pos) -> List[Pos]:
    return astar([list(r) for r in grid], s, g) if HAS_AI else _bfs(grid, s, g)

def _adv(grid, s: Pos, g: Pos, steps: int) -> Pos:
    p = _path(grid, s, g)
    return p[min(steps, len(p)-1)] if p else s

def _intercept(grid, monster: Pos, human: Pos, offset: int = 4) -> Pos:
    h, w = len(grid), len(grid[0])
    dr = human[0]-monster[0]; dc = human[1]-monster[1]
    if abs(dr) > h//2: dr = dr-h if dr>0 else dr+h
    if abs(dc) > w//2: dc = dc-w if dc>0 else dc+w
    sr = (1 if dr>0 else -1) if dr else 0
    sc = (1 if dc>0 else -1) if dc else 0
    t = wrap((human[0]+sr*offset, human[1]+sc*offset), grid)
    return t if passable(grid, t) else human

def ai_astar(grid, human: Pos, monsters: List[Pos]) -> Tuple[List[Pos], float]:
    t0 = time.perf_counter()
    return [_adv(grid, monsters[0], human, 2)], time.perf_counter()-t0

def ai_greedy(grid, human: Pos, monsters: List[Pos]) -> Tuple[List[Pos], float]:
    if not HAS_AI: return ai_astar(grid, human, monsters)
    t0 = time.perf_counter()
    ctrl = make_greedy_controller(
        walls_from_grid(grid), width=len(grid[0]), height=len(grid),
        steps_per_turn=2, allow_stay=False, avoid_stacking=True)
    paths = ctrl.decide((human[1], human[0]), many_row_col_to_xy(monsters))
    return list(many_xy_to_row_col(p[-1] for p in paths)), time.perf_counter()-t0

def ai_minimax(grid, human: Pos, monsters: List[Pos]) -> Tuple[List[Pos], float]:
    if not HAS_AI: return ai_astar(grid, human, monsters)
    t0 = time.perf_counter()
    ctrl = make_minimax_controller(
        walls_from_grid(grid), width=len(grid[0]), height=len(grid),
        steps_per_turn=2, depth=2, player_can_stay=False, monster_can_stay=False)
    paths = ctrl.decide((human[1], human[0]), many_row_col_to_xy(monsters))
    return list(many_xy_to_row_col(p[-1] for p in paths)), time.perf_counter()-t0

def ai_sa(
    grid,
    human: Pos,
    monsters: List[Pos],
    sa_previous_move: Optional[Tuple[Pos, Pos]] = None,
) -> Tuple[List[Pos], float]:
    if not HAS_AI: return ai_astar(grid, human, monsters)
    t0 = time.perf_counter()
    try:
        # run_cpp_map_algorithm 内部已处理临时文件的读写与执行，直接传入当前的元组坐标即可
        _, m = run_cpp_map_algorithm(
            "sa",
            grid,
            human,
            monsters[0],
            sa_previous_move=sa_previous_move,
        )
        return [m], time.perf_counter()-t0
    except Exception as e:
        print(f"[WARN] SA 算法执行失败 ({e})，回退到 A* 移动")
        return [_adv(grid, monsters[0], human, 2)], time.perf_counter()-t0

# ════════════════════════════════════════════════════════════════════════════
# Simulation
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class GameResult:
    steps: int
    caught: bool
    step_times: List[float]

def simulate(grid, human: Pos, monsters: List[Pos], fn) -> GameResult:
    H, W = len(grid), len(grid[0])
    step_times: List[float] = []
    sa_previous_move: Optional[Tuple[Pos, Pos]] = None
    for step in range(1, MAX_STEPS+1):
        if human in monsters:
            return GameResult(step-1, True, step_times)
        d = player_move(grid, human, monsters)
        if d is None:
            return GameResult(step, False, step_times)
        dr, dc = VALID_DIRECTIONS[d]
        nxt = ((human[0]+dr)%H, (human[1]+dc)%W)
        if passable(grid, nxt): human = nxt
        if human in monsters:
            return GameResult(step, True, step_times)
        previous_monsters = monsters
        if fn is ai_sa:
            new_m, dt = fn(
                grid,
                human,
                monsters,
                sa_previous_move=sa_previous_move,
            )
        else:
            new_m, dt = fn(grid, human, monsters)
        step_times.append(dt)
        monsters = new_m
        if (
            fn is ai_sa
            and previous_monsters
            and monsters
            and previous_monsters[0] != monsters[0]
        ):
            sa_previous_move = (previous_monsters[0], monsters[0])
        if human in monsters:
            return GameResult(step, True, step_times)
    return GameResult(MAX_STEPS, False, step_times)

def evaluate(name: str, fn, maps: List[Tuple]) -> Dict[str, Any]:
    print(f"  [{name}] {len(maps)} games ...", flush=True)
    results: List[GameResult] = []
    for i, (grid, human, monster) in enumerate(maps):
        results.append(simulate(grid, human, [monster], fn))
        if (i+1) % 10 == 0:
            print(f"    {i+1}/{len(maps)}", flush=True)
    sa = np.array([r.steps for r in results], dtype=float)
    ta = np.array([t*1000 for r in results for t in r.step_times] or [0.0])
    return {
        "name": name,
        "steps_mean":    float(np.mean(sa)),
        "steps_median":  float(np.median(sa)),
        "steps_std":     float(np.std(sa)),
        "steps_all":     sa.tolist(),
        "time_mean_ms":  float(np.mean(ta)),
        "time_median_ms":float(np.median(ta)),
        "time_std_ms":   float(np.std(ta)),
        "times_all_ms":  ta.tolist(),
        "caught_rate":   float(np.mean([r.caught for r in results])),
    }

# ════════════════════════════════════════════════════════════════════════════
# Chart style constants
# ════════════════════════════════════════════════════════════════════════════

COLORS = {
    "A*":     "#2563eb",
    "Greedy": "#16a34a",
    "Minimax":"#9333ea",
    "SA":     "#ea580c",
}
BG      = "#f0f4f8"
AX_BG   = "#ffffff"
GRID_C  = "#dde3ea"
SPINE_C = "#b0bac4"
TXT_DK  = "#1e293b"
TXT_MU  = "#64748b"

def _new_fig(w=8, h=5.8):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX_BG)
    return fig, ax

def _style(ax, xlabel="", ylabel=""):
    ax.spines[["top","right"]].set_visible(False)
    ax.spines["left"].set_color(SPINE_C)
    ax.spines["bottom"].set_color(SPINE_C)
    ax.tick_params(colors=TXT_MU, labelsize=10)
    ax.yaxis.grid(True, color=GRID_C, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel, color=TXT_MU, fontsize=10.5, labelpad=7)
    if ylabel: ax.set_ylabel(ylabel, color=TXT_MU, fontsize=10.5, labelpad=7)

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {path}")

# ════════════════════════════════════════════════════════════════════════════
# Six independent charts
# ════════════════════════════════════════════════════════════════════════════

def chart1_steps_bar(stats, out):
    fig, ax = _new_fig()
    fig.suptitle("Avg Steps to Catch Player",
                 fontsize=15, fontweight="bold", color=TXT_DK, y=0.98)
    names  = [s["name"] for s in stats]
    means  = [s["steps_mean"] for s in stats]
    stds   = [s["steps_std"]  for s in stats]
    colors = [COLORS[n] for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, means, color=colors, width=0.48, zorder=3,
                  yerr=stds, capsize=7,
                  error_kw={"ecolor":TXT_MU,"lw":1.5,"capthick":1.5})
    for bar, v, sd in zip(bars, means, stds):
        ax.text(bar.get_x()+bar.get_width()/2, v+sd+1.8,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=TXT_DK)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12, color=TXT_DK)
    ax.text(0.98, 0.97, "Lower = stronger",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color=TXT_MU, style="italic")
    _style(ax, ylabel="Steps")
    _save(fig, os.path.join(out, "fig1_steps_bar.png"))

def chart2_steps_box(stats, out):
    fig, ax = _new_fig()
    fig.suptitle("Steps Distribution (Box Plot)",
                 fontsize=15, fontweight="bold", color=TXT_DK, y=0.98)
    names  = [s["name"] for s in stats]
    data   = [s["steps_all"] for s in stats]
    colors = [COLORS[n] for n in names]
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True, widths=0.42,
                    medianprops={"color":"white","lw":2.5},
                    whiskerprops={"color":SPINE_C,"lw":1.4},
                    capprops={"color":SPINE_C,"lw":1.4},
                    flierprops={"marker":"o","markersize":4,
                                "markerfacecolor":"#cbd5e1",
                                "markeredgecolor":"#94a3b8",
                                "linestyle":"none","alpha":0.7})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.88)
    ax.tick_params(axis="x", labelsize=12, colors=TXT_DK)
    _style(ax, ylabel="Steps")
    _save(fig, os.path.join(out, "fig2_steps_box.png"))

def chart3_catch_rate(stats, out):
    fig, ax = _new_fig(w=7.5, h=5)
    fig.suptitle("Catch Rate within Step Limit",
                 fontsize=15, fontweight="bold", color=TXT_DK, y=0.98)
    names  = [s["name"] for s in stats]
    rates  = [s["caught_rate"]*100 for s in stats]
    colors = [COLORS[n] for n in names]
    y = np.arange(len(names))
    hb = ax.barh(y, rates, color=colors, height=0.42, zorder=3)
    for bar, v in zip(hb, rates):
        ax.text(v+0.9, bar.get_y()+bar.get_height()/2,
                f"{v:.1f}%", va="center", fontsize=11,
                fontweight="bold", color=TXT_DK)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=12, color=TXT_DK)
    ax.set_xlim(0, 115)
    ax.xaxis.grid(True, color=GRID_C, lw=0.9, zorder=0)
    ax.yaxis.grid(False)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.spines["bottom"].set_color(SPINE_C)
    ax.tick_params(axis="y", left=False, colors=TXT_MU, labelsize=10)
    ax.set_xlabel("Catch Rate (%)", color=TXT_MU, fontsize=10.5, labelpad=7)
    ax.set_axisbelow(True)
    _save(fig, os.path.join(out, "fig3_catch_rate.png"))

def chart4_time_bar(stats, out):
    fig, ax = _new_fig()
    fig.suptitle("Avg Decision Time per Step",
                 fontsize=15, fontweight="bold", color=TXT_DK, y=0.98)
    names  = [s["name"] for s in stats]
    tmeans = [s["time_mean_ms"] for s in stats]
    tstds  = [s["time_std_ms"]  for s in stats]
    colors = [COLORS[n] for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, tmeans, color=colors, width=0.48, zorder=3,
                  yerr=tstds, capsize=7,
                  error_kw={"ecolor":TXT_MU,"lw":1.5,"capthick":1.5})
    for bar, v, sd in zip(bars, tmeans, tstds):
        ax.text(bar.get_x()+bar.get_width()/2,
                v+sd+max(v*0.04, 0.003),
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=TXT_DK)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12, color=TXT_DK)
    ax.text(0.98, 0.97, "Lower = faster",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color=TXT_MU, style="italic")
    _style(ax, ylabel="Time (ms)")
    _save(fig, os.path.join(out, "fig4_time_bar.png"))

def chart5_time_box(stats, out):
    fig, ax = _new_fig()
    fig.suptitle("Decision Time Distribution  (log scale)",
                 fontsize=15, fontweight="bold", color=TXT_DK, y=0.98)
    names  = [s["name"] for s in stats]
    data   = [[max(t,1e-5) for t in s["times_all_ms"]] for s in stats]
    colors = [COLORS[n] for n in names]
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True, widths=0.42,
                    medianprops={"color":"white","lw":2.5},
                    whiskerprops={"color":SPINE_C,"lw":1.4},
                    capprops={"color":SPINE_C,"lw":1.4},
                    flierprops={"marker":"o","markersize":3.5,
                                "markerfacecolor":"#cbd5e1",
                                "markeredgecolor":"#94a3b8",
                                "linestyle":"none","alpha":0.6})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.88)
    ax.set_yscale("log")
    ax.yaxis.grid(True, color=GRID_C, which="both", lw=0.9, zorder=0)
    ax.tick_params(axis="x", labelsize=12, colors=TXT_DK)
    _style(ax, ylabel="Time (ms, log scale)")
    _save(fig, os.path.join(out, "fig5_time_box.png"))

def chart6_overview(stats, out):
    fig, ax = _new_fig(w=8, h=6)
    fig.suptitle("Performance Overview  ·  Steps vs Decision Time",
                 fontsize=15, fontweight="bold", color=TXT_DK, y=0.98)
    for s in stats:
        c = COLORS[s["stone"]] if "stone" in s else COLORS[s["name"]]
        ax.scatter(s["steps_mean"], s["time_mean_ms"],
                   s=240, color=c, zorder=5,
                   edgecolors="white", linewidths=2.2)
        ax.errorbar(s["steps_mean"], s["time_mean_ms"],
                    xerr=s["steps_std"], yerr=s["time_std_ms"],
                    fmt="none", ecolor=c, alpha=0.35, lw=1.6, capsize=5)
        ax.annotate(s["name"],
                    (s["steps_mean"], s["time_mean_ms"]),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=12, fontweight="bold", color=c)
    _style(ax,
           xlabel="Avg Steps to Catch  (lower = stronger AI)",
           ylabel="Avg Decision Time (ms)  (lower = faster)")
    ax.xaxis.grid(True, color=GRID_C, lw=0.9, zorder=0)
    ax.text(0.03, 0.05,
            "Ideal: bottom-left\n(strongest & fastest)",
            transform=ax.transAxes, fontsize=9, color=TXT_MU, va="bottom",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor=AX_BG, edgecolor=GRID_C, alpha=0.92))
    _save(fig, os.path.join(out, "fig6_overview.png"))

# ════════════════════════════════════════════════════════════════════════════
# Text report
# ════════════════════════════════════════════════════════════════════════════

def print_report(stats):
    SEP = "=" * 74
    print(f"\n{SEP}")
    print(f"  Algorithm Evaluation Report")
    print(SEP)
    print(f"  {'Algo':<10} {'AvgSteps':>9} {'Median':>8} {'Std':>7}"
          f"  {'Catch%':>7}  {'AvgTime(ms)':>12}  {'MedTime(ms)':>12}")
    print("-"*74)
    for s in stats:
        print(f"  {s['name']:<10}"
              f" {s['steps_mean']:>9.1f}"
              f" {s['steps_median']:>8.1f}"
              f" {s['steps_std']:>7.1f}"
              f"  {s['caught_rate']*100:>6.1f}%"
              f"  {s['time_mean_ms']:>12.4f}"
              f"  {s['time_median_ms']:>12.4f}")
    print(SEP)
    bs = min(stats, key=lambda s: s["steps_mean"])
    bt = min(stats, key=lambda s: s["time_mean_ms"])
    print(f"\n  Strongest: {bs['name']} (avg {bs['steps_mean']:.1f} steps)")
    print(f"  Fastest  : {bt['name']} (avg {bt['time_mean_ms']:.4f} ms/step)\n")

# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    global _ga_params
    parser = argparse.ArgumentParser(description="Monster AI Evaluation")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--skip",  nargs="*", default=[])
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--out",   default="eval_charts")
    parser.add_argument("--fast",  action="store_true")
    args = parser.parse_args()

    if args.fast:
        _ga_params = GA_FAST
        print("[INFO] Fast mode: lightweight GA params")

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    all_algos = [
        ("A*",      ai_astar),
        ("Greedy",  ai_greedy),
        ("Minimax", ai_minimax),
        ("SA",      ai_sa),
    ]
    skip_set = {s.lower() for s in (args.skip or [])}
    algos = [(n, fn) for n, fn in all_algos if n.lower() not in skip_set]
    if not algos:
        print("No algorithms left."); return

    # ── Pre-generate all maps (shared by all algorithms) ─────────────────
    print(f"\n{'='*55}")
    print(f"  Generating {args.games} maps (genetic algorithm) ...")
    print(f"  GA params: {_ga_params}")
    print(f"{'='*55}")
    t0 = time.time()
    maps = []
    for i in range(args.games):
        maps.append(get_map())
        if (i+1) % 5 == 0:
            print(f"  Map {i+1}/{args.games} done", flush=True)
    print(f"  Maps ready in {time.time()-t0:.1f}s\n")

    # ── Evaluate each algorithm on the same maps ──────────────────────────
    print(f"{'='*55}")
    print(f"  Evaluating {len(algos)} algorithms × {args.games} games")
    print(f"{'='*55}\n")
    stats = []
    for name, fn in algos:
        s = evaluate(name, fn, maps)
        stats.append(s)
        print(f"  --> AvgSteps={s['steps_mean']:.1f}  "
              f"Catch={s['caught_rate']*100:.0f}%  "
              f"AvgTime={s['time_mean_ms']:.4f}ms\n")

    print_report(stats)

    # ── Save 6 independent charts ─────────────────────────────────────────
    print(f"Saving 6 charts to '{args.out}/' ...")
    chart1_steps_bar(stats, args.out)
    chart2_steps_box(stats, args.out)
    chart3_catch_rate(stats, args.out)
    chart4_time_bar(stats, args.out)
    chart5_time_box(stats, args.out)
    chart6_overview(stats, args.out)
    print("\nAll done.")

if __name__ == "__main__":
    main()
