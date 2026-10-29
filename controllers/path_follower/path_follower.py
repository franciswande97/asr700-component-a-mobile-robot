"""
path_follower.py
Component A, Phase 4: closed-loop autonomous navigation.

Integrates the three subsystems built in Phases 1 to 3:
  Perception : LiDAR + camera fusion. When both sensors agree an obstacle is
               close, the robot slows down.
  Planning   : follows the A* waypoints saved by path_planning.py
               (planned_path.csv, in world coordinates).
  Control    : PID heading controller (gains tuned in pid_tuning.py) plus a
               speed schedule (slow when turning sharply or near the goal).

Localisation: GPS gives (x, y), the inertial unit gives the heading.
The run is logged to trajectory_log.csv for evaluate_run.py.
"""
import csv
import math
import os

import cv2
import numpy as np
from controller import Robot

HERE = os.path.dirname(os.path.abspath(__file__))
WAYPOINT_FILE = os.path.join(HERE, "planned_path.csv")
LOG_FILE = os.path.join(HERE, "trajectory_log.csv")

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
# Robot (same values as pid_tuning.py)
WHEEL_RADIUS = 0.0975         # m
AXLE_LENGTH = 0.33            # m
MAX_WHEEL_SPEED = 5.24        # rad/s
MAX_OMEGA = 2 * WHEEL_RADIUS * MAX_WHEEL_SPEED / AXLE_LENGTH

# PID heading controller (gains chosen in pid_tuning.py)
KP, KI, KD = 4.0, 3.0, 0.6
I_LIMIT = 1.0
I_ZONE = math.radians(20)     # only integrate when the heading error is small

# Navigation
CRUISE_SPEED = 0.25           # m/s
WAYPOINT_TOLERANCE = 0.20     # m, switch to the next waypoint inside this distance
GOAL_TOLERANCE = 0.10         # m, goal reached inside this distance
SLOW_RADIUS = 0.6             # m, start slowing this far from the final goal
MAX_RUN_TIME = 120.0          # s, safety limit
YAW_SIGN = 1                  # change to -1 if the robot turns away from the path

# Perception (LiDAR + camera fusion from Phase 1)
OBSTACLE_RANGE_THRESHOLD = 0.5   # m
OBSTACLE_EDGE_THRESHOLD = 50     # camera edge pixels
SLOW_FACTOR = 0.5                # speed multiplier when the fusion detects an obstacle


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, low, high):
    return max(low, min(high, value))


def load_waypoints(path):
    with open(path) as f:
        return [(float(row["x_m"]), float(row["y_m"])) for row in csv.DictReader(f)]


def wheel_speeds(v, omega):
    """Forward speed (m/s) and turn rate (rad/s) -> wheel speeds (rad/s)."""
    w_left = (v - omega * AXLE_LENGTH / 2) / WHEEL_RADIUS
    w_right = (v + omega * AXLE_LENGTH / 2) / WHEEL_RADIUS
    scale = max(1.0, abs(w_left) / MAX_WHEEL_SPEED, abs(w_right) / MAX_WHEEL_SPEED)
    return w_left / scale, w_right / scale      # keep the ratio if a motor would saturate


# ----------------------------------------------------------------------------
# Devices
# ----------------------------------------------------------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0

gps = robot.getDevice("gps")
imu = robot.getDevice("inertial unit")
camera = robot.getDevice("camera")
lidar = robot.getDevice("lidar")
for device in (gps, imu, camera, lidar):
    device.enable(timestep)
width, height = camera.getWidth(), camera.getHeight()

left_motor = robot.getDevice("left wheel")
right_motor = robot.getDevice("right wheel")
for motor in (left_motor, right_motor):
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)

waypoints = load_waypoints(WAYPOINT_FILE)
last_idx = len(waypoints) - 1
target_idx = 1 if last_idx > 0 else 0       # waypoint 0 is the start cell
print(f"Loaded {len(waypoints)} waypoints, goal = {waypoints[-1]}")

log_file = open(LOG_FILE, "w", newline="")
log = csv.writer(log_file)
log.writerow(["t", "x", "y", "yaw", "heading_error", "dist_to_waypoint", "waypoint_idx",
              "closest_range", "edge_count", "fused_obstacle", "v", "omega"])

integral = 0.0
prev_yaw = None
step_count = 0
printed_start = False

# ----------------------------------------------------------------------------
# Main loop: perceive -> plan (waypoint) -> control
# ----------------------------------------------------------------------------
while robot.step(timestep) != -1:
    t = robot.getTime()
    x, y, _ = gps.getValues()
    yaw = YAW_SIGN * imu.getRollPitchYaw()[2]
    if not printed_start:
        print(f"Start pose: x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f} deg")
        printed_start = True

    # --- Perception: LiDAR + camera fusion ---
    ranges = lidar.getRangeImage()
    closest_range = min(ranges) if ranges else float("inf")
    img = np.frombuffer(camera.getImage(), np.uint8).reshape((height, width, 4))
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)
    edge_count = int(np.count_nonzero(cv2.Canny(gray, 50, 150)))
    fused_obstacle = (closest_range < OBSTACLE_RANGE_THRESHOLD
                      and edge_count > OBSTACLE_EDGE_THRESHOLD)

    # --- Planning: pick the current waypoint ---
    while True:
        wx, wy = waypoints[target_idx]
        dist = math.hypot(wx - x, wy - y)
        is_final = target_idx == last_idx
        if not is_final and dist < WAYPOINT_TOLERANCE:
            target_idx += 1
            integral = 0.0                      # fresh integral for the new leg
            print(f"t={t:5.1f}s  -> waypoint {target_idx}/{last_idx}")
        else:
            break

    if is_final and dist < GOAL_TOLERANCE:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        log.writerow([round(t, 3), x, y, yaw, 0.0, dist, target_idx, closest_range,
                      edge_count, int(fused_obstacle), 0.0, 0.0])
        print(f"GOAL REACHED at t={t:.1f} s, final distance to goal {dist:.3f} m")
        break

    # --- Control: PID on heading error ---
    desired_heading = math.atan2(wy - y, wx - x)
    error = wrap(desired_heading - yaw)
    if abs(error) < I_ZONE:
        integral = clamp(integral + error * dt, -I_LIMIT, I_LIMIT)
    derivative = 0.0 if prev_yaw is None else -wrap(yaw - prev_yaw) / dt
    prev_yaw = yaw
    omega = clamp(KP * error + KI * integral + KD * derivative, -MAX_OMEGA, MAX_OMEGA)

    # --- Speed schedule ---
    speed = CRUISE_SPEED * max(0.0, math.cos(error))     # turn on the spot if badly misaligned
    if is_final:
        speed *= min(1.0, dist / SLOW_RADIUS)            # ease in to the goal
    if fused_obstacle:
        speed *= SLOW_FACTOR                             # perception slows the robot down

    w_left, w_right = wheel_speeds(speed, omega)
    left_motor.setVelocity(w_left)
    right_motor.setVelocity(w_right)

    log.writerow([round(t, 3), x, y, yaw, error, dist, target_idx, closest_range,
                  edge_count, int(fused_obstacle), speed, omega])
    step_count += 1
    if step_count % 20 == 0:
        log_file.flush()

    if t > MAX_RUN_TIME:
        print(f"Stopped: run time limit of {MAX_RUN_TIME:.0f} s reached, "
              f"distance to waypoint {dist:.2f} m")
        break

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)
log_file.close()
print(f"Log saved to {LOG_FILE}")