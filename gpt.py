#!/usr/bin/env python3
"""
Fixed arm simulator using solve_ivp with proper event handling and units.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

## -----------------------
## PARAMETERS (SI units)
## -----------------------
# convert mm -> m for geometry
mm = 1e-3

m_arm  =    0.92         # kg
d11    =    45.0 * mm    # m
d12    =   269.0 * mm    # m
d21    =    45.0 * mm    # m
d22    =   240.0 * mm    # m
x1t    =   100.0 * mm    # m
x2t    =   123.0 * mm    # m
x1_0   =   146.0 * mm    # m
x2_0   =   155.0 * mm    # m
l_arm  =   340.0 * mm    # m (unused)
d_arm  =   150.0 * mm    # m (center of mass distance)
d_load =   330.0 * mm    # m
r_arm  =    35.0 * mm    # m
l_1    =     0.0         # m (bicep length placeholder)
l_2    =     0.0         # m (tricep length placeholder)

J      =  0.0279         # kg·m^2 (inertia)
G      =  9.80665        # m/s^2 (gravity)

## fitted parameters
b_arm =  0.221           # N·m·s/rad (rotational damping)
km1   =    207.0         # N/m (bicep spring)
bm1   = 0.0395           # N·s/m (bicep damping)
km2   =    126.0         # N/m (tricep spring)
bm2   =   1.99           # N·s/m (tricep damping)
klim  = 0.0361           # N·m/rad (limiter spring)
blim  =   6e-3           # N·m·s/rad (limiter damping)

# joint limits (radians)
θlim1 =  2.618           # upper (≈150°)
θlim2 =  0.873           # lower (≈50°)

## actuator caps (user-provided guesses)
BICEP_MAX_FORCE   = 1500.0   # N
TRICEP_MAX_FORCE  =  600.0   # N

m_load = 0.0                 # kg (nothing in hand)

# small eps to avoid re-triggering events at same time
TIME_EPS = 1e-12

## -----------------------
## RHS (dynamics)
## -----------------------
def RHS(t, state, bicep_motor, tricep_motor):
    """
    state: [theta, dtheta]
    motors: values in [0,1]
    returns: [dtheta, ddtheta]
    """
    θ, dθ = state

    # Geometry: tendon lengths (safe)
    # l1 is distance between two points that depend on theta.
    # formula used originally: l1 = sqrt(d11**2 + d12**2 - 2*d11*d12*cos(theta))
    # ensure it's not zero
    cosθ = np.cos(θ)
    sinθ = np.sin(θ)
    l1_sq = d11**2 + d12**2 - 2.0 * d11 * d12 * cosθ
    l1 = np.sqrt(max(l1_sq, 1e-12))

    # x1 is tendon extension relative to some rest length x1t
    x1 = l1 - x1t
    # dl1/dθ = d11*d12*sin(θ)/l1
    dl1_dθ = (d11 * d12 * sinθ) / l1
    dotx1 = dl1_dθ * dθ

    # For the second tendon the original code used a (d22^2 - d21^2)**0.5 + d21*(pi - θ)
    # keep same form but convert to safe operations
    # NOTE: this is geometry-specific and kept as in your original
    base_l2 = np.sqrt(max(d22**2 - d21**2, 0.0))
    l2 = base_l2 + d21 * (np.pi - θ)
    x2 = l2 - x2t
    dotx2 = -d21 * dθ

    # Muscle forces (simple model)
    F1 = (BICEP_MAX_FORCE * bicep_motor) + bm1 * dotx1 + km1 * (x1 - x1_0)
    F2 = (TRICEP_MAX_FORCE * tricep_motor) + bm2 * dotx2 + km2 * (x2 - x2_0)

    # muscles cannot pull negative (no active push)
    F1 = max(F1, 0.0)
    F2 = max(F2, 0.0)

    # limiter torques (only when pushing into the limit)
    t_lim1 = 0.0
    if θ > θlim1 and dθ > 0.0:
        t_lim1 = -klim * (θ - θlim1) - blim * dθ

    t_lim2 = 0.0
    if θ < θlim2 and dθ < 0.0:
        t_lim2 = -klim * (θ - θlim2) - blim * dθ

    # gravity torques: restoring sign -> -m g L sin(theta)
    gravity_on_lower_arm = - m_arm * d_arm * G * sinθ
    gravity_on_load      = - m_load * d_load * G * sinθ

    # moment arms: F1 acts through dl1/dθ (force times derivative of length -> torque)
    # torque contribution from F1 = F1 * (dl1/dθ)
    # torque contribution from F2 = - F2 * d21   (as in your original)
    τ_total = (t_lim1 + t_lim2
               + F1 * dl1_dθ
               - F2 * d21
               + gravity_on_lower_arm
               + gravity_on_load
               - b_arm * dθ * 10.0)   # existing scale in original

    ddθ = τ_total / J

    return [dθ, ddθ]


## -----------------------
## EVENTS (only position limits)
## -----------------------
def event_theta_lim1(t, state, b1, b2):
    θ, dθ = state
    return θ - θlim1
event_theta_lim1.terminal = True
event_theta_lim1.direction = +1   # only trigger when θ increases past θlim1

def event_theta_lim2(t, state, b1, b2):
    θ, dθ = state
    return θ - θlim2
event_theta_lim2.terminal = True
event_theta_lim2.direction = -1   # only trigger when θ decreases past θlim2

# Note: we removed the dθ==0 event because it's not a physical discontinuity


## -----------------------
## Arm class with hybrid integration (proper resets)
## -----------------------
class Arm:
    def __init__(self):
        self.t = 0.0
        self.θ = np.pi * 0.8   # starting angle
        self.dθ = 0.0

    def next_state_would_be(self, dt, motors=(0.0, 0.0)):
        exit_time = self.t + dt
        temp_t = self.t
        temp_θ = self.θ
        temp_dθ = self.dθ

        # Ensure motors is a tuple of two numbers for args
        motors_tuple = tuple(motors)

        while temp_t < exit_time - 1e-15:
            sol = solve_ivp(
                RHS,
                [temp_t, exit_time],
                [temp_θ, temp_dθ],
                args=motors_tuple,
                events=[event_theta_lim1, event_theta_lim2],
                method='RK45',
                max_step=1e-4,
                rtol=1e-8,
                atol=1e-9
            )
            # take last state
            temp_t = sol.t[-1]
            temp_θ, temp_dθ = sol.y[0, -1], sol.y[1, -1]

            # If an event occurred, detect which one and reset/clip
            events = sol.t_events  # list of arrays, same order as events list
            event_fired = None
            for idx, ev_times in enumerate(events):
                if ev_times.size > 0:
                    event_fired = idx
                    break

            if event_fired is not None:
                # event 0 -> θlim1, event 1 -> θlim2
                if event_fired == 0:
                    # clamp angle at upper limit, zero velocity (sticky limit)
                    temp_θ = θlim1
                    temp_dθ = 0.0
                elif event_fired == 1:
                    temp_θ = θlim2
                    temp_dθ = 0.0

                # advance time by a tiny epsilon to avoid immediate re-trigger
                temp_t = temp_t + TIME_EPS

            # if no event, loop will exit when temp_t == exit_time

        return temp_θ, temp_dθ

    def step(self, dt, motors=(0.0, 0.0)):
        next_θ, next_dθ = self.next_state_would_be(dt, motors)
        self.t += dt
        self.θ = next_θ
        self.dθ = next_dθ


## -----------------------
## MAIN: run a short sim and plot
## -----------------------
if __name__ == "__main__":
    arm = Arm()
    dt = 0.001
    times = []
    angles = []
    angular_velocities = []

    # Example: hold bicep fully on, tricep off
    motors = (1.0, 0.0)

    while arm.t < 0.5 - 1e-12:
        arm.step(dt, motors=motors)
        times.append(arm.t)
        angles.append(arm.θ)
        angular_velocities.append(arm.dθ)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(times, angles, label="angle (rad)")
    plt.yticks([0, np.pi/2, np.pi], ['0 (up)', 'π/2 (hor.)', 'π (down)'])
    plt.ylim(0, np.pi * 1.2)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(times, angular_velocities, label="angular velocity (rad/s)")
    plt.legend()
    plt.xlabel("time (s)")

    plt.tight_layout()
    plt.show()
