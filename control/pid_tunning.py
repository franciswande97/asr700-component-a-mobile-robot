"""
pid_tuning.py
Component A, Phase 3: kinematic model and PID controller tuning.

The robot is a differential-drive vehicle. Its pose is (x, y, theta) and it is
driven by a forward speed v and a turn rate omega:

    x_dot     = v * cos(theta)
    y_dot     = v * sin(theta)
    theta_dot = omega

Left/right wheel angular speeds (rad/s) follow from v and omega:

    w_left  = (v - omega * L / 2) / R
    w_right = (v + omega * L / 2) / R

with R = wheel radius and L = distance between the wheels.

A PID controller turns the heading error (desired heading minus actual heading)
into the turn rate omega. This script tests that controller on a heading step
of 90 degrees with six different gain sets, and measures overshoot, rise time,
settling time and steady-state error for each.

Run:  python pid_tuning.py
Outputs go to a "results" folder next to this script.
"""
import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
# 1. ROBOT AND SIMULATION SETTINGS
# ----------------------------------------------------------------------------
WHEEL_RADIUS = 0.0975        # m, Pioneer 3-DX wheel radius (assumed from the datasheet)
AXLE_LENGTH = 0.33           # m, distance between the wheels (assumed)
MAX_WHEEL_SPEED = 5.24       # rad/s, Pioneer 3-DX motor limit in Webots
MAX_OMEGA = 2 * WHEEL_RADIUS * MAX_WHEEL_SPEED / AXLE_LENGTH   # turn rate when spinning on the spot

DT = 0.032                   # s, control step (set to your Webots basicTimeStep / 1000)
ACTUATOR_LAG = 0.15          # s, first-order lag of the wheels reaching the commanded speed
DELAY_STEPS = 2              # control steps of sensing/actuation delay
DISTURBANCE = 0.08           # rad/s, constant yaw drift (wheel mismatch, slip)
I_LIMIT = 1.0                # anti-windup: hard limit on the integral term
I_ZONE = math.radians(20)    # anti-windup: only integrate when the error is within 20 degrees

STEP_TARGET = math.radians(90)   # heading step: turn from 0 to 90 degrees
T_END = 8.0                      # s
BAND = 0.02                      # settling band: +/- 2 % of the step

# (name, Kp, Ki, Kd)
GAIN_SETS = [
    ("P low (Kp=1)",         1.0, 0.0, 0.0),
    ("P medium (Kp=3)",      3.0, 0.0, 0.0),
    ("P too high (Kp=25)",  25.0, 0.0, 0.0),
    ("PD (3, 0, 0.5)",       3.0, 0.0, 0.5),
    ("PI (3, 1.5, 0)",       3.0, 1.5, 0.0),
    ("PID (4, 3, 0.6)",      4.0, 3.0, 0.6),
]
CHOSEN = "PID (4, 3, 0.6)"       # the set we carry into the Webots controller (Phase 4)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def wrap(angle):
    """Wrap an angle to the range (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, low, high):
    return max(low, min(high, value))


# ----------------------------------------------------------------------------
# 2. KINEMATIC MODEL
# ----------------------------------------------------------------------------
def wheel_speeds(v, omega):
    """Forward speed v (m/s) and turn rate omega (rad/s) -> wheel speeds (rad/s)."""
    w_left = (v - omega * AXLE_LENGTH / 2) / WHEEL_RADIUS
    w_right = (v + omega * AXLE_LENGTH / 2) / WHEEL_RADIUS
    return w_left, w_right


# ----------------------------------------------------------------------------
# 3. PID HEADING CONTROLLER ON THE MODEL
# ----------------------------------------------------------------------------
def simulate_heading_step(kp, ki, kd):
    """Simulate the closed loop and return time, heading (rad) and turn rate (rad/s)."""
    n_steps = int(T_END / DT)
    theta = 0.0
    omega_actual = 0.0
    integral = 0.0
    prev_theta = theta
    command_queue = [0.0] * DELAY_STEPS      # models the sensing/actuation delay
    times, headings, omegas = [], [], []

    for k in range(n_steps):
        error = wrap(STEP_TARGET - theta)
        if abs(error) < I_ZONE:                                      # anti-windup: integral separation
            integral = clamp(integral + error * DT, -I_LIMIT, I_LIMIT)
        derivative = -wrap(theta - prev_theta) / DT                  # derivative on measurement
        prev_theta = theta

        omega_cmd = clamp(kp * error + ki * integral + kd * derivative, -MAX_OMEGA, MAX_OMEGA)

        command_queue.append(omega_cmd)
        omega_delayed = command_queue.pop(0)
        omega_actual += (omega_delayed - omega_actual) * DT / (ACTUATOR_LAG + DT)

        theta += (omega_actual + DISTURBANCE) * DT

        times.append(k * DT)
        headings.append(theta)
        omegas.append(omega_actual)

    return np.array(times), np.array(headings), np.array(omegas)


# ----------------------------------------------------------------------------
# 4. RESPONSE METRICS
# ----------------------------------------------------------------------------
def response_metrics(t, theta):
    target = STEP_TARGET
    overshoot = max(0.0, (theta.max() - target) / target * 100.0)

    t10 = t[np.argmax(theta >= 0.1 * target)] if np.any(theta >= 0.1 * target) else math.nan
    t90 = t[np.argmax(theta >= 0.9 * target)] if np.any(theta >= 0.9 * target) else math.nan
    rise_time = t90 - t10

    outside = np.abs(theta - target) > BAND * target
    if not outside.any():
        settling_time = 0.0
    elif outside[-1]:
        settling_time = math.nan                 # never settles inside the band
    else:
        settling_time = t[np.nonzero(outside)[0][-1] + 1]

    last_second = t >= (t[-1] - 1.0)
    steady_state_error = math.degrees(np.mean(target - theta[last_second]))
    return overshoot, rise_time, settling_time, steady_state_error


def fmt(value, digits=2):
    return "never" if (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


# ----------------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Wheel speeds for v=0.3 m/s, omega=0.5 rad/s: "
          f"{tuple(round(w, 2) for w in wheel_speeds(0.3, 0.5))} rad/s")
    print(f"Maximum turn rate on the spot: {MAX_OMEGA:.2f} rad/s\n")

    header = f"{'Gain set':<20}{'Overshoot %':>13}{'Rise (s)':>10}{'Settle (s)':>12}{'SS error (deg)':>16}"
    print(header)
    print("-" * len(header))

    fig, (ax_theta, ax_err) = plt.subplots(1, 2, figsize=(15, 5.5))
    rows = []
    for name, kp, ki, kd in GAIN_SETS:
        t, theta, _ = simulate_heading_step(kp, ki, kd)
        overshoot, rise, settle, sse = response_metrics(t, theta)
        print(f"{name:<20}{overshoot:>13.1f}{fmt(rise):>10}{fmt(settle):>12}{sse:>16.2f}")
        rows.append([name, kp, ki, kd, round(overshoot, 2), fmt(rise), fmt(settle), round(sse, 3)])
        ax_theta.plot(t, np.degrees(theta), label=name, linewidth=1.8)
        ax_err.plot(t, np.degrees(STEP_TARGET - theta), label=name, linewidth=1.8)

    target_deg = math.degrees(STEP_TARGET)
    ax_theta.axhline(target_deg, color="black", linestyle="--", linewidth=1, label="target (90°)")
    ax_theta.axhspan(target_deg * (1 - BAND), target_deg * (1 + BAND), color="grey", alpha=0.2,
                     label="±2 % band")
    ax_theta.set_title("Heading response to a 90° step")
    ax_theta.set_xlabel("time (s)")
    ax_theta.set_ylabel("heading (deg)")
    ax_theta.legend(fontsize=8)
    ax_theta.grid(alpha=0.3)

    ax_err.axhline(0, color="black", linewidth=1)
    ax_err.set_title("Heading error over time")
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("error (deg)")
    ax_err.set_ylim(-60, 95)
    ax_err.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "phase3_step_responses.png"), dpi=200)

    with open(os.path.join(RESULTS_DIR, "phase3_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gain_set", "Kp", "Ki", "Kd", "overshoot_percent", "rise_time_s",
                         "settling_time_s", "steady_state_error_deg"])
        writer.writerows(rows)

    print(f"\nChosen gains for the Webots controller: {CHOSEN}")
    print(f"Saved plot and metrics to: {RESULTS_DIR}")
    plt.show()


if __name__ == "__main__":
    main()