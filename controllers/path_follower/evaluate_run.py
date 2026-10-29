"""
evaluate_run.py
Component A, Phase 4: compare the planned path with the path the robot drove.

Reads planned_path.csv and trajectory_log.csv (same folder as this script),
prints a performance table, and saves trajectory_plot.png and run_metrics.csv.

Run:  python evaluate_run.py
"""
import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
PLANNED_FILE = os.path.join(HERE, "planned_path.csv")
LOG_FILE = os.path.join(HERE, "trajectory_log.csv")

# Same world description as path_planning.py (for drawing the map only)
ARENA_SIZE = (6.0, 6.0)
ARENA_CENTER = (0.0, 0.0)
OBSTACLES_WORLD = [                 # (centre_x, centre_y, size_x, size_y)
    (-1.48831, -0.352735, 0.3, 0.6),
    (1.3, -1.0, 0.6, 0.6),
]
GOAL_REACHED_DISTANCE = 0.15       # m


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def distance_to_polyline(points, polyline):
    """Shortest distance from each (x, y) point to a polyline of waypoints."""
    best = np.full(len(points), np.inf)
    for a, b in zip(polyline[:-1], polyline[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0) if denom > 0 else np.zeros(len(points))
        closest = a + t[:, None] * ab
        best = np.minimum(best, np.linalg.norm(points - closest, axis=1))
    return best


def main():
    planned_data = read_csv(PLANNED_FILE)
    planned = np.column_stack([planned_data["x_m"], planned_data["y_m"]])
    run = read_csv(LOG_FILE)
    t = run["t"]
    actual = np.column_stack([run["x"], run["y"]])

    cte = distance_to_polyline(actual, planned)                   # cross-track error (m)
    planned_length = float(np.sum(np.linalg.norm(np.diff(planned, axis=0), axis=1)))
    actual_length = float(np.sum(np.linalg.norm(np.diff(actual, axis=0), axis=1)))
    final_error = float(np.linalg.norm(actual[-1] - planned[-1]))
    reached = final_error <= GOAL_REACHED_DISTANCE
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    slowed_time = float(np.sum(run["fused_obstacle"]) * step)

    metrics = [
        ("Goal reached", "yes" if reached else "no"),
        ("Time to goal (s)", f"{t[-1] - t[0]:.1f}"),
        ("Planned path length (m)", f"{planned_length:.2f}"),
        ("Actual path length (m)", f"{actual_length:.2f}"),
        ("Path length ratio (actual / planned)", f"{actual_length / planned_length:.3f}"),
        ("Mean cross-track error (m)", f"{cte.mean():.3f}"),
        ("RMS cross-track error (m)", f"{math.sqrt(np.mean(cte ** 2)):.3f}"),
        ("Max cross-track error (m)", f"{cte.max():.3f}"),
        ("Final distance to goal (m)", f"{final_error:.3f}"),
        ("Time slowed by perception (s)", f"{slowed_time:.1f}"),
        ("Closest LiDAR reading (m)", f"{run['closest_range'].min():.2f}"),
    ]
    width = max(len(name) for name, _ in metrics)
    print("\nPerformance evaluation")
    print("-" * (width + 12))
    for name, value in metrics:
        print(f"{name:<{width}}  {value}")

    with open(os.path.join(HERE, "run_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics)

    # --- plots ---
    fig, (ax_map, ax_cte, ax_head) = plt.subplots(1, 3, figsize=(18, 5.5),
                                                   gridspec_kw={"width_ratios": [1.3, 1, 1]})
    cx, cy = ARENA_CENTER
    ax_map.add_patch(Rectangle((cx - ARENA_SIZE[0] / 2, cy - ARENA_SIZE[1] / 2),
                               ARENA_SIZE[0], ARENA_SIZE[1], fill=False, edgecolor="black",
                               linewidth=2))
    for ox, oy, sx, sy in OBSTACLES_WORLD:
        ax_map.add_patch(Rectangle((ox - sx / 2, oy - sy / 2), sx, sy, color="saddlebrown",
                                   alpha=0.8))
    ax_map.plot(planned[:, 0], planned[:, 1], "r--", linewidth=2, marker="o", markersize=5,
                label="planned path (A*)")
    ax_map.plot(actual[:, 0], actual[:, 1], color="royalblue", linewidth=2, label="actual path")
    ax_map.scatter(*planned[0], c="green", s=100, zorder=5, label="start")
    ax_map.scatter(*planned[-1], c="gold", s=180, marker="*", edgecolors="black", zorder=5,
                   label="goal")
    ax_map.set_aspect("equal")
    ax_map.set_title("Planned vs actual trajectory")
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    ax_map.legend(fontsize=8, loc="upper left")
    ax_map.grid(alpha=0.3)

    ax_cte.plot(t, cte, color="darkorange")
    ax_cte.set_title("Cross-track error over time")
    ax_cte.set_xlabel("time (s)")
    ax_cte.set_ylabel("distance from planned path (m)")
    ax_cte.grid(alpha=0.3)

    ax_head.plot(t, np.degrees(run["heading_error"]), color="purple")
    ax_head.axhline(0, color="black", linewidth=1)
    ax_head.set_title("Heading error over time")
    ax_head.set_xlabel("time (s)")
    ax_head.set_ylabel("error (deg)")
    ax_head.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "trajectory_plot.png"), dpi=200)
    print(f"\nSaved trajectory_plot.png and run_metrics.csv to {HERE}")
    plt.show()


if __name__ == "__main__":
    main()