"""
perception_controller.py
Pioneer 3-DX: LiDAR + camera fusion, plus sonar-based avoidance.

The sonar scoring and turn state machine are adapted from the Webots
Pioneer 3-DX obstacle avoidance sample controller (Cyberbotics Ltd,
Apache License 2.0), ported from C to Python.
"""
from controller import Robot
import numpy as np
import cv2

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# --- Sensors ---
camera = robot.getDevice("camera")
lidar = robot.getDevice("lidar")
camera.enable(timestep)
lidar.enable(timestep)
lidar.enablePointCloud()
width, height = camera.getWidth(), camera.getHeight()

# The robot's 16 built-in sonars: so0 ... so15
sonars = []
for i in range(16):
    s = robot.getDevice(f"so{i}")
    s.enable(timestep)
    sonars.append(s)

# --- Motors ---
left_motor = robot.getDevice("left wheel")
right_motor = robot.getDevice("right wheel")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# --- Tunable settings ---
FORWARD_SPEED = 3.0              # rad/s, cruising
TURN_SPEED = 0.7 * 5.24          # rad/s, spin-in-place (0.7 x Pioneer max speed)
OBSTACLE_RANGE_THRESHOLD = 0.5   # metres, used by LiDAR AND sonar
OBSTACLE_EDGE_THRESHOLD = 50     # camera edge pixels

# Sonar scoring (from the Webots sample)
SONAR_MAX_VALUE = 1024           # raw sensor value at 0 m
SONAR_MAX_RANGE = 5.0            # metres, range covered by the sensor's lookup table
SCORE_THRESHOLD = 100            # score above this means "blocked"
# (left_score, right_score) weights per sonar. so0-so3 face the left side,
# so4-so7 face the right side, and the front sonars weigh the most.
SONAR_WEIGHTS = [(150, 0), (200, 0), (300, 0), (600, 0),
                 (0, 600), (0, 300), (0, 200), (0, 150)] + [(0, 0)] * 8

FORWARD, TURNING_RIGHT, TURNING_LEFT = "FORWARD", "TURNING_RIGHT", "TURNING_LEFT"
state = FORWARD

while robot.step(timestep) != -1:
    # --- Camera -> edge count ---
    raw = camera.getImage()
    img = np.frombuffer(raw, np.uint8).reshape((height, width, 4))
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)
    edge_count = int(np.count_nonzero(cv2.Canny(gray, 50, 150)))

    # --- LiDAR closest reading ---
    ranges = lidar.getRangeImage()
    closest_range = min(ranges) if ranges else float('inf')

    # --- Fusion: both sensors must agree ---
    fused_obstacle = (closest_range < OBSTACLE_RANGE_THRESHOLD
                      and edge_count > OBSTACLE_EDGE_THRESHOLD)

    # --- Sonar scores: how blocked is the left side vs the right side? ---
    left_score, right_score = 0.0, 0.0
    for i, s in enumerate(sonars):
        value = s.getValue()
        if value == 0.0:          # sees nothing within range
            continue
        distance = SONAR_MAX_RANGE * (1.0 - value / SONAR_MAX_VALUE)
        if distance < OBSTACLE_RANGE_THRESHOLD:
            modifier = 1.0 - distance / OBSTACLE_RANGE_THRESHOLD  # closer = stronger
            left_score += SONAR_WEIGHTS[i][0] * modifier
            right_score += SONAR_WEIGHTS[i][1] * modifier
    sonar_obstacle = left_score > SCORE_THRESHOLD or right_score > SCORE_THRESHOLD

    blocked = fused_obstacle or sonar_obstacle

    # --- State machine ---
    previous_state = state
    if state == FORWARD:
        if blocked:
            # Obstacle more on the left -> spin right, otherwise spin left.
            # (If only fusion fired, the scores are equal, so it defaults to right.)
            state = TURNING_RIGHT if left_score >= right_score else TURNING_LEFT
    else:
        if not blocked:           # keep turning the same way until clear
            state = FORWARD

    if state == FORWARD:
        left_speed, right_speed = FORWARD_SPEED, FORWARD_SPEED
    elif state == TURNING_RIGHT:  # left wheel forward, right wheel back
        left_speed, right_speed = TURN_SPEED, -TURN_SPEED
    else:                         # TURNING_LEFT
        left_speed, right_speed = -TURN_SPEED, TURN_SPEED

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    # --- Log only on state changes ---
    if state != previous_state:
        reason = "fusion" if fused_obstacle else "sonar"
        if fused_obstacle and sonar_obstacle:
            reason = "fusion+sonar"
        print(f"{previous_state} -> {state} ({reason}) "
              f"closest={closest_range:.2f} m, edges={edge_count}, "
              f"sonar L={left_score:.0f} R={right_score:.0f}")