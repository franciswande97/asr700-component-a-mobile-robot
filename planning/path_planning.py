"""
path_planning.py
Component A, Phase 2: motion planning and pathfinding.

Pipeline
  1. Build an occupancy grid of the arena (1 = occupied, 0 = free).
  2. Inflate obstacles by the robot radius so the robot is treated as a point.
  3. Run Dijkstra and A* (two heuristics) on the same grid.
  4. Compare them: nodes expanded, nodes generated, path length, runtime.
  5. Save plots, a metrics table and the planned waypoints (used in Phase 3/4).

Run:  python path_planning.py
Outputs go to a "results" folder next to this script.
"""
import csv
import heapq
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# ----------------------------------------------------------------------------
# 1. MAP DEFINITION  
# ----------------------------------------------------------------------------
CELL_SIZE = 0.1                   # metres per grid cell
ROBOT_RADIUS_CELLS = 3            # ceil(0.267 m swing radius / 0.1 m) -> 0.3 m margin
ARENA_SIZE = (6.0, 6.0)           # RectangleArena floorSize (x, y), metres
ARENA_CENTER = (0.0, 0.0)         # RectangleArena translation (x, y)

# Obstacles in WORLD metres: (centre_x, centre_y, size_x, size_y)
OBSTACLES_WORLD = [
    (-1.48831, -0.352735, 0.3, 0.6),
    (1.3, -1, 0.6, 0.6)
]
START_WORLD = (-2.2, -0.35)        # the robot's translation (x, y)
GOAL_WORLD = (2.2, -0.35)           # any free spot you choose

COLS = round(ARENA_SIZE[0] / CELL_SIZE)
ROWS = round(ARENA_SIZE[1] / CELL_SIZE)
WORLD_OFFSET = ARENA_CENTER       # world position of the grid centre


def world_to_cell(x, y):
    """World (x, y) in metres -> grid (row, col)."""
    col = math.floor((x - (WORLD_OFFSET[0] - COLS * CELL_SIZE / 2)) / CELL_SIZE)
    row = math.floor(((WORLD_OFFSET[1] + ROWS * CELL_SIZE / 2) - y) / CELL_SIZE)
    return row, col


def world_rect_to_cells(cx, cy, sx, sy):
    """Box in world metres -> (row_start, col_start, row_end, col_end), inclusive."""
    r0, c0 = world_to_cell(cx - sx / 2, cy + sy / 2)                    # top-left corner
    r1, c1 = world_to_cell(cx + sx / 2 - 1e-9, cy - sy / 2 + 1e-9)      # bottom-right corner
    return r0, c0, r1, c1


# Outer ring of cells = the arena walls, plus the obstacles converted from metres.
OBSTACLE_RECTS = [
    (0, 0, 0, COLS - 1),                # top wall
    (ROWS - 1, 0, ROWS - 1, COLS - 1),  # bottom wall
    (0, 0, ROWS - 1, 0),                # left wall
    (0, COLS - 1, ROWS - 1, COLS - 1),  # right wall
] + [world_rect_to_cells(*o) for o in OBSTACLES_WORLD]

START = world_to_cell(*START_WORLD)    # (row, col)
GOAL = world_to_cell(*GOAL_WORLD)      # (row, col)

SQRT2 = math.sqrt(2.0)
# 8-connected moves: (d_row, d_col, step_cost)
MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TIMING_REPEATS = 30          # searches are fast, so average the runtime


def build_grid():
    grid = np.zeros((ROWS, COLS), dtype=np.uint8)
    for r0, c0, r1, c1 in OBSTACLE_RECTS:
        grid[r0:r1 + 1, c0:c1 + 1] = 1
    return grid


def inflate(grid, radius):
    """Grow every obstacle by `radius` cells so the robot can be planned as a point."""
    inflated = grid.copy()
    rows_idx, cols_idx = np.nonzero(grid)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr = np.clip(rows_idx + dr, 0, grid.shape[0] - 1)
            cc = np.clip(cols_idx + dc, 0, grid.shape[1] - 1)
            inflated[rr, cc] = 1
    return inflated


# ----------------------------------------------------------------------------
# 2. HEURISTICS
# ----------------------------------------------------------------------------
def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def zero(a, b):
    return 0.0


# ----------------------------------------------------------------------------
# 3. SEARCH  (Dijkstra is A* with a heuristic that is always 0)
# ----------------------------------------------------------------------------
def _search(grid, start, goal, heuristic):
    rows, cols = grid.shape
    g_cost = {start: 0.0}
    parent = {}
    open_heap = [(heuristic(start, goal), start)]   # entries: (f = g + h, node)
    closed = set()
    explored_order = []
    generated = 1
    found = False

    while open_heap:
        _, node = heapq.heappop(open_heap)
        if node in closed:           # stale heap entry, a cheaper one was used already
            continue
        closed.add(node)
        explored_order.append(node)
        if node == goal:
            found = True
            break
        r, c = node
        for dr, dc, step in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr, nc]:
                continue
            # forbid cutting the corner of an obstacle on a diagonal move
            if dr and dc and (grid[r + dr, c] or grid[r, c + dc]):
                continue
            nxt = (nr, nc)
            if nxt in closed:
                continue
            new_g = g_cost[node] + step
            if new_g < g_cost.get(nxt, math.inf):
                g_cost[nxt] = new_g
                parent[nxt] = node
                heapq.heappush(open_heap, (new_g + heuristic(nxt, goal), nxt))
                generated += 1

    path = []
    if found:
        node = goal
        while node != start:
            path.append(node)
            node = parent[node]
        path.append(start)
        path.reverse()

    return {"found": found, "path": path,
            "cost": g_cost.get(goal, math.inf),
            "expanded": len(explored_order), "generated": generated,
            "explored_order": explored_order}


def dijkstra(grid, start, goal):
    return _search(grid, start, goal, zero)


def astar(grid, start, goal, heuristic=euclidean):
    return _search(grid, start, goal, heuristic)


def timed(label, heuristic_name, fn, repeats=TIMING_REPEATS):
    """Run a search `repeats` times and attach the mean runtime in milliseconds."""
    runtimes = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        runtimes.append((time.perf_counter() - t0) * 1000.0)
    result["label"] = label
    result["heuristic"] = heuristic_name
    result["runtime_ms"] = float(np.mean(runtimes))
    return result


# ----------------------------------------------------------------------------
# 4. PATH POST-PROCESSING
# ----------------------------------------------------------------------------
def simplify(path):
    """Keep only the cells where the direction changes (waypoints)."""
    if len(path) < 3:
        return path
    waypoints = [path[0]]
    for prev, cur, nxt in zip(path, path[1:], path[2:]):
        if (cur[0] - prev[0], cur[1] - prev[1]) != (nxt[0] - cur[0], nxt[1] - cur[1]):
            waypoints.append(cur)
    waypoints.append(path[-1])
    return waypoints


def cell_to_world(cell):
    """Grid cell (row, col) -> world (x, y) in metres. +x right, +y up, grid centred."""
    r, c = cell
    x = (c + 0.5) * CELL_SIZE - COLS * CELL_SIZE / 2 + WORLD_OFFSET[0]
    y = (ROWS / 2 - r - 0.5) * CELL_SIZE + WORLD_OFFSET[1]
    return x, y


# ----------------------------------------------------------------------------
# 5. VISUALISATION
# ----------------------------------------------------------------------------
def plot_search(ax, raw, inflated, res, start, goal):
    display = inflated.astype(int) + raw.astype(int)       # 0 free, 1 margin, 2 obstacle
    ax.imshow(display, cmap=ListedColormap(["white", "#cfcfcf", "black"]),
              origin="upper", vmin=0, vmax=2)

    explored = np.zeros(raw.shape)
    for r, c in res["explored_order"]:
        explored[r, c] = 1
    ax.imshow(np.ma.masked_where(explored == 0, explored), cmap="Blues",
              alpha=0.55, origin="upper", vmin=0, vmax=2)

    if res["found"]:
        ys, xs = zip(*res["path"])
        ax.plot(xs, ys, color="red", linewidth=2.2, label="planned path")
    ax.scatter([start[1]], [start[0]], c="green", s=90, marker="o", label="start", zorder=5)
    ax.scatter([goal[1]], [goal[0]], c="gold", s=130, marker="*", edgecolors="black",
               label="goal", zorder=5)
    ax.set_title(f"{res['label']}\nexpanded {res['expanded']} | "
                 f"length {res['cost'] * CELL_SIZE:.2f} m | {res['runtime_ms']:.2f} ms",
                 fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_comparison(results, path):
    labels = [r["label"] for r in results]
    colours = ["#4c72b0", "#dd8452", "#55a868"]
    metrics = [("Nodes expanded", [r["expanded"] for r in results]),
               ("Path length (m)", [r["cost"] * CELL_SIZE for r in results]),
               ("Mean runtime (ms)", [r["runtime_ms"] for r in results])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (title, values) in zip(axes, metrics):
        bars = ax.bar(range(len(values)), values, color=colours[:len(values)])
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_title(title)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}" if v < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Dijkstra vs A*: efficiency and optimality", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


# ----------------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    raw = build_grid()
    inflated = inflate(raw, ROBOT_RADIUS_CELLS)
    assert not inflated[START] and not inflated[GOAL], "Start or goal is inside an obstacle"

    results = [
        timed("Dijkstra", "none (h = 0)", lambda: dijkstra(inflated, START, GOAL)),
        timed("A* (Euclidean)", "Euclidean", lambda: astar(inflated, START, GOAL, euclidean)),
        timed("A* (Manhattan)", "Manhattan", lambda: astar(inflated, START, GOAL, manhattan)),
    ]
    best = min(r["cost"] for r in results)

    # --- metrics table ---
    header = f"{'Algorithm':<16}{'Expanded':>10}{'Generated':>11}{'Length (m)':>12}" \
             f"{'Gap vs best':>13}{'Runtime (ms)':>14}"
    print(header)
    print("-" * len(header))
    rows_out = []
    for r in results:
        length_m = r["cost"] * CELL_SIZE
        gap = (r["cost"] - best) / best * 100.0
        print(f"{r['label']:<16}{r['expanded']:>10}{r['generated']:>11}{length_m:>12.2f}"
              f"{gap:>12.1f}%{r['runtime_ms']:>14.3f}")
        rows_out.append([r["label"], r["heuristic"], r["found"], r["expanded"], r["generated"],
                         round(length_m, 3), round(gap, 2), round(r["runtime_ms"], 4)])
    with open(os.path.join(RESULTS_DIR, "phase2_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "heuristic", "path_found", "nodes_expanded", "nodes_generated",
                    "path_length_m", "gap_vs_best_percent", "mean_runtime_ms"])
        w.writerows(rows_out)

    # --- path plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, res in zip(axes, results):
        plot_search(ax, raw, inflated, res, START, GOAL)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Occupancy grid search: explored cells (blue), inflated margin (grey), "
                 "obstacles (black)", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(os.path.join(RESULTS_DIR, "phase2_paths.png"), dpi=200)

    plot_comparison(results, os.path.join(RESULTS_DIR, "phase2_comparison.png"))

    # --- planned waypoints for the controller (Phase 3 / 4) ---
    astar_result = results[1]
    waypoints = simplify(astar_result["path"])
    with open(os.path.join(RESULTS_DIR, "planned_path.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_m", "y_m"])
        for cell in waypoints:
            x, y = cell_to_world(cell)
            w.writerow([round(x, 4), round(y, 4)])
    print(f"\nA* path: {len(astar_result['path'])} cells reduced to {len(waypoints)} waypoints")
    print(f"Saved plots, metrics and waypoints to: {RESULTS_DIR}")

    plt.show()


if __name__ == "__main__":
    main()