import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------------------
# Physical parameters
# ---------------------------
g = 9.81 # gravity
e = 0.8 # elsaticity of bounce
bounce_height = 0.2   # <-- PLATFORM HEIGHT
y0 = 1.0 # initial height
v0 = 0.0 # initial velocity
t0 = 0.0
tf = 10
eps = 1e-8
max_bounces = 10

# ---------------------------
# ODE system
# ---------------------------
def dynamics(t, state):
    y, v = state # height, velocity
    return [v, -g]

# ---------------------------
# Event: hit platform at y = 0.2
# ---------------------------
def hit_platform(t, state):
    return state[0] - bounce_height # when this is negative, then, the event fires
hit_platform.terminal = True

# def woohoo(t, state):
#     return state[0] - 0.5
# woohoo.terminal = False
# # hit_platform.direction = -1

# ---------------------------
# Simulation loop
# ---------------------------
t_current = t0
state_current = np.array([y0, v0])
bounces = 0

t_all = []
y_all = []
v_all = []

impact_times = []
other_event_times = []

while t_current < tf and bounces < max_bounces:

    sol = solve_ivp(
        dynamics,
        (t_current, tf),
        state_current,
        events=[hit_platform],
        max_step=0.02,
        rtol=1e-7,
        atol=1e-9
    )

    t_all.append(sol.t)
    y_all.append(sol.y[0])
    v_all.append(sol.y[1])

    # print(sol.t_events)

    # If impact detected
    if sol.t_events != None and sol.t_events[0].size > 0:

        t_impact = sol.t_events[0][0]
        v_pre = sol.y[1, -1]

        # Discontinuous velocity reset
        v_post = -e * v_pre

        impact_times.append(t_impact)
        bounces += 1

        # Restart slightly above platform
        t_current = t_impact + eps
        state_current = np.array([bounce_height + v_post * eps, v_post])
    
    # elif sol.t_events != None and sol.t_events[1].size > 0:
    #     # print("SOMETHING")
    #     other_event_times.append(sol.t_events[1][0])
    #     state_current = [0.5, -sol.y[1, -1]]
    #     continue

    else:
        break
    

# ---------------------------
# Combine data
# ---------------------------
t_all = np.concatenate(t_all)
y_all = np.concatenate(y_all)
v_all = np.concatenate(v_all)

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(10, 4.5))
plt.plot(t_all, y_all)
plt.axhline(bounce_height, linestyle="--")  # platform height
plt.scatter(impact_times, [bounce_height]*len(impact_times), marker="x")
plt.scatter(other_event_times, [0.5]*len(other_event_times), marker="x")
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Ball Bouncing on a Platform at y = 0.2 m")
plt.grid(True)
plt.tight_layout()
plt.show()
