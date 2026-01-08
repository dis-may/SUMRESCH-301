from pylab import *
from scipy.integrate import solve_ivp
import numpy as np


## ARM PARAMETERS
m_arm  =    0.92 ## kg
d11    =    0.045 ## m
d12    =    0.269 ## m
d21    =    0.045 ## m 
d22    =   0.240 ## m
x1t    =   0.100 ## m
x2t    =   0.123 ## m
x1_0   =   0.146 ## m
x2_0   =   0.155 ## m
l_arm  =   0.340 ## m
d_arm  =   0.150 ## m        
d_load =   0.330 ## m (we use this as the length asw)
r_arm  =   0.035 ## m        
l_1    =     0.0 ## bicep length (mm)
l_2    =     0.0 ## tricep length (mm)
J      =  0.0279 ## kg.m^2 (nothing in hand) 
G      =  9.80665 ## m/s^2

arm_height = 1.0 ## m (the height of the elbow joint above the ground)

## BALL PARAMETERS
e = 0.8 # elasticity
r_ball = 0.12 ## m (average radius of basketball)
x_ball = d_load ## m (the distance of the arm = fixed ball x pos)


## fitted parameters
b_arm =  0.221  ## Nms/rad (damping coefficient)
km1   =    207  ## N/m (bicep spring constant?)
bm1   = 0.0395  ## Ns/m (bicep damping coefficient?)
km2   =    126  ## N/m (tricep spring constant?)
bm2   =   1.99  ## Ns/m (tricep damping coefficient?)
klim  = 0.0361  ## Nm/rad (limiter spring constant?)
blim  =   6E-3  ## Nms/rad (limiter damping coefficient?)

θlim1 =  2.618  ## rad (150 degrees)
θlim2 =  0.873  ## rad (50 degrees)

## FOR ME TO DETERMINE
BICEP_MAX_FORCE = 1500.0 ## N (google AI generated guess)
TRICEP_MAX_FORCE = 600.0 ## N (google AI generated guess)
eps = 1E-8 # epsilon to advance time when event is triggered

m_load = 0.0 ## kg (nothing in hand)

def RHS(time, state, bicep_motor, tricep_motor) :
    θ, dθ, y, v = state
    # print(f'RHS: {θ=}, {dθ=}')
    ## motors should be between 0 and 1

    l1 = np.sqrt( d11**2 + d12**2 - 2*d11*d12*np.cos(θ) )
    assert l1 > 0.0, "l1 is non-positive!"+f" {l1=}, {d11=}, {d12=}, {θ=}"
    x1 = l1 - x1t
    dotx1 = (abs(d11**2 - d12**2 - 2*d11*d12*np.cos(θ))**-0.5) * d11*d12 * np.sin(θ) * dθ # double check why abs is there
    
    l2 = np.sqrt( d22**2 - d21**2 ) + d21*(np.pi - θ)
    x2 = l2 - x2t
    dotx2 = -d21 * dθ

    
    F1 = (BICEP_MAX_FORCE  * bicep_motor)  + bm1 * dotx1 + km1 * (x1 - x1_0)
    F2 = (TRICEP_MAX_FORCE * tricep_motor) + bm2 * dotx2 + km2 * (x2 - x2_0)

    if F1 < 0 :
        F1 = 0.0
    if F2 < 0 :
        F2 = 0.0    
    
    t_lim1 = 0.0
    if θ > θlim1 :
        if dθ > 0 :
            t_lim1 = -klim * (θ - θlim1) - blim * dθ
        else :
            t_lim1 = -klim * (θ - θlim1)

    t_lim2 = 0.0
    if θ < θlim2 :
        if dθ < 0 :
            t_lim2 = -klim * (θ - θlim2) - blim * dθ
        else :
            t_lim2 = -klim * (θ - θlim2)

    gravity_on_lower_arm = m_arm  * d_arm  * G * np.sin(θ) 
    gravity_on_load      = m_load * d_load * G * np.sin(θ)

    # print(f"{time=}, {θ=}, {dθ=}, {F1=}, {F2=}, {t_lim1=}, {t_lim2=}, {gravity_on_lower_arm=}, {gravity_on_load=}")

    ddθ = (1.0/J) * (t_lim1 + t_lim2
                    - F1 * ((d11*d12 * np.sin(θ)/l1)) 
                    + F2 * d21 
                    + gravity_on_lower_arm 
                    + gravity_on_load
                    - b_arm * dθ 
                    )
    
    ## θ, dθ  <-- State variables
    return [dθ, ddθ, v, -G] ## return derivatives

### discontinuities

def discontinuity_θlim1(t, state, bicep_motor, tricep_motor): 
    θ, dθ, y, v = state
    return θ - θlim1
discontinuity_θlim1.terminal=True ## when the event occurs, stop
# discontinuity_θlim1.direction = +1 

def discontinuity_θlim2(t, state, bicep_motor, tricep_motor): 
    θ, dθ, y, v = state
    return θ - θlim2
discontinuity_θlim2.terminal=True ## when the event occurs, stop
# discontinuity_θlim2.direction = -1 

def hit_platform(t, state, bicep_motor, tricep_motor):
    θ, dθ, y, v = state
    return y - r_ball # when this is negative, then, the event fires (height is 0)
hit_platform.terminal = True

def collision(t, state, bicep_motor, tricep_motor):
    # average height 
    θ, dθ, y, v = state

    # coordinates of the arm
    x_arm = sin(θ) * d_load
    y_arm = cos(θ) * d_load + arm_height# d_load is the arm radius

    # x_ball is fixed
    y_ball = y

    # length of the vector of the difference between arm and ball
    distance = np.sqrt((x_arm - x_ball)**2 + (y_arm - y_ball)**2)
    if distance < r_ball:
        return -1.0
    else:
        return 1
hit_platform.terminal = False # change back to True later

class ArmBall:
    def __init__(self, θ=np.pi/2) :
        ## VARIABLES
        self.t = 0.0    ## time
        self.θ  =  θ ## arm_angle; pi = down towards gravity; pi/2 = horizontal
        self.dθ =  0.0    ## arm_angular_velocity 
        self.y = 1.0 ## initial height
        self.v = 0.0 ## initial velocity
        self.event_times = []   
        self.event_values = [] 
        self.collision_times = []
        self.collision_values = []
                             
    def next_state_would_be(self, dt, motors=[0.0,0.0]) :
        exit_time = self.t + dt
        temp_t  = self.t
        temp_θ  = self.θ
        temp_dθ = self.dθ        
        temp_y = self.y
        temp_v = self.v

        # print(f"--- Integrating from {temp_t} to {exit_time} ---")

        while temp_t < exit_time :
            sol = solve_ivp(RHS, [temp_t, exit_time], [temp_θ, temp_dθ, temp_y, temp_v], args=(motors),
                            dense_output=True, 
                            events=[
                                discontinuity_θlim1,
                                discontinuity_θlim2,
                                hit_platform,
                                collision,
                                ], 
                            method='RK45', max_step=0.01)
            temp_t = sol.t[-1]            
            temp_θ, temp_dθ, temp_y, temp_v = sol.y[0,-1], sol.y[1,-1], sol.y[2,-1], sol.y[3,-1]            
            # print(f"{temp_t=}, {temp_θ=}, {temp_dθ=}")
            #quit()
            # print(f"  Events: {sol.t_events}")
            # quit()

            # If an event occurred, detect which one and reset/clip
            events = sol.t_events  # list of arrays, same order as events list
            event_fired = None
            for idx, ev_times in enumerate(events):
                if ev_times.size > 0:
                    event_fired = idx
                    break

            if event_fired == 0 or event_fired == 1:
                # when a discontinuity is reached, terminal = True
                # print("EVENT FIRED", sol.t[-1])
                self.event_times.append(sol.t[-1])
                self.event_values.append(θlim1 if idx == 0 else θlim2)

                # reset the integration by starting it with the next step
                dθ, ddθ, v, g = RHS(sol.t[-1], [temp_θ, temp_dθ, temp_y, temp_v], motors[0], motors[1])
                temp_θ = temp_θ + (dθ * eps)
                temp_dθ = temp_dθ + (ddθ * eps)
                temp_y = temp_y + (v * eps)
                temp_v = temp_v + (g * eps)

                # advance time by a tiny epsilon to avoid immediate re-trigger
                temp_t = temp_t + eps
            
            elif event_fired == 2: # ball touches the ground
                t_impact = sol.t_events[2][0]
                v_pre = sol.y[3, -1]

                # Discontinuous velocity reset
                temp_v = -e * v_pre

                # Restart slightly above platform
                temp_t = t_impact + eps
                temp_y = r_ball + temp_v * eps
            
            elif event_fired == 3: # collision
                self.collision_times.append(sol.t[-1])
                self.collision_values.append(temp_y)
                print("SD:LFHSDKLJF")


            
            
        return temp_θ, temp_dθ, temp_y, temp_v
    
    def step(self, dt,  motors=[0.0,0.0]) :
        next_θ, next_dθ, next_y, next_v = self.next_state_would_be(dt, motors)
        self.t  += dt
        self.θ   = next_θ
        self.dθ  = next_dθ
        self.y = next_y
        self.v = next_v


def initial_plotting_setup():
    arm = ArmBall()
    dt = 0.001
    times = []
    angles = []
    ball_y = []
    ball_v = []
    angular_velocities = []
    bicep = 0.00
    tricep = 0.00
    
    while arm.t < 3:
        # arm.step(dt, motors=[0.00,0.005]) # looks nice
        arm.step(dt, motors=[bicep, tricep])
        times.append(arm.t)
        angles.append(arm.θ)
        angular_velocities.append(arm.dθ)
        ball_y.append(arm.y)
        ball_v.append(arm.v)

    subplot2grid((2,1), (0,0))
    title(f"Bicep={bicep}, Tricep={tricep}, normal ")
    heights = [cos(angle) * d_arm + arm_height for angle in angles]
    plot(times, heights, label="angle (rad)")
    
    # plot(arm.event_times, arm.event_values, marker='x', linestyle=None)
    # plt.axhline(y=θlim1, color='r', linestyle='--', linewidth=1) 
    # plt.axhline(y=θlim2, color='r', linestyle='--', linewidth=1) 
    ax = plt.gca() # Get the current axes
    # ax.yaxis.set_inverted(True) # Invert the y-axis
    # yticks([0, θlim2, np.pi/2, θlim1, np.pi], ['0 (up)', 'θlim2', 'π/2 (hor.)',  'θlim1', 'π (down)'])
    # ylim(0, np.pi*1.2)
    legend()
    subplot2grid((2,1), (1,0))
    # plot(times, angular_velocities, label="angular velocity (rad/s)")
    plot(times, ball_y, label="height (y) in metres")
    plot(arm.collision_times, arm.collision_values, marker='x', linestyle=None)

    legend()

    # savefig("./plots/b0_t0_005_normal_sign")
    
    show()




if __name__ == "__main__" :
    duration = 3
    dt = 0.001

    initial_plotting_setup()

