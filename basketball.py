

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -----------------------------
# Physical parameters
# -----------------------------
m = 0.62          # mass of basketball (kg)
g = 9.81          # gravity (m/s^2)

k = 1.5e5         # Hertz stiffness (N/m^(3/2))
alpha = 0.6       # Hunt–Crossley dissipation parameter (s/m)
#alpha = 0.2       # Hunt–Crossley dissipation parameter (s/m)

# Ground / hand position (fixed)
def x_hand(t):
    return 0.0

def v_hand(t):
    return 0.0

# -----------------------------
# Dynamics
# -----------------------------
def dynamics(t, y):
    x, v = y

    delta = x_hand(t) - x
    delta_dot = v_hand(t) - v

    F_contact = 0.0
    if delta > 0:
        F_contact = k * delta**1.5 * (1 + alpha * delta_dot)
        if F_contact < 0:   # no tensile contact
            F_contact = 0.0

    dxdt = v
    dvdt = -g + F_contact / m

    return [dxdt, dvdt]

# -----------------------------
# Initial conditions
# -----------------------------
y0 = [0.2, 0.0]   # height (m), velocity (m/s)

# -----------------------------
# Integration
# -----------------------------
t_span = (0, 1.0)
t_eval = np.linspace(*t_span, 3000)

sol = solve_ivp(
    dynamics,
    t_span,
    y0,
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10,
    method='Radau'
)

t = sol.t
x = sol.y[0]
v = sol.y[1]

# -----------------------------
# Compute contact force history
# -----------------------------
F_hand = np.zeros_like(t)

for i in range(len(t)):
    delta = x_hand(t[i]) - x[i]
    delta_dot = v_hand(t[i]) - v[i]

    if delta > 0:
        F = k * delta**1.5 * (1 + alpha * delta_dot)
        F_hand[i] = max(F, 0.0)   # force applied to hand (N)

# -----------------------------
# Plot results
# -----------------------------
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.ylabel("Height (m)")
plt.title("Basketball with Hunt–Crossley Contact")

plt.subplot(3, 1, 2)
plt.plot(t, v)
plt.ylabel("Velocity (m/s)")

plt.subplot(3, 1, 3)
plt.plot(t, F_hand)
plt.xlabel("Time (s)")
plt.ylabel("Force on hand (N)")

plt.tight_layout()
plt.show()
