from pylab import *

class Genotype:
    def __init__(self, *args):
        ## e.g. ls_lm=None, lm_rm=None, rs_lm=None, rs_rm=None
        self.genes = list(args)
        self.n = len(args)

    def randomise(self, gene_min=-1, gene_max=1):
        for i in range(self.n):
            self.genes[i] = np.random.uniform(gene_min, gene_max)


class Light:
    def __init__(self, x=0, y=0, intensity=1):
        self.model = None
        
        self.x = x
        self.y = y
        self.intensity = intensity # multiplier for how bright the light source
    
    def __repr__(self):
        return f"Light(d x={self.x}, y={self.y})"


class Vehicle:
    def __init__(self, genes:Genotype, x, y, a, r_wheel=1.0, r_robot=0.1, b=np.pi/4, light=Light(0,0)):
        ## STATIC PARAMETERS
        self.genes = genes # [ls_lm, ls_rm, rs_lm, rs_rm]
        self.r_wheel = r_wheel # radius of the wheel
        self.r_robot = r_robot # radius of the robot
        self.b = b # the angle between the directional sensors
        self.light = light # assume only one light 

        ## DYNAMIC VARIABLES
        self.x = x
        self.y = y
        self.a = a # direction the robot is facing

        self.dxdt = 0
        self.dydt = 0
        self.dadt = 0
    
    def __repr__(self):
        return f"Vehicle(x={self.x}, y={self.y}, a={self.a}, r={self.r_wheel}, d={self.r_robot})"
    
    def get_state(self) -> tuple[int, int, int]:
        # for plotting/recording purposes
        return (self.x, self.y, self.a,)

    def get_sensor_positions(self) -> tuple[float, float, float, float, float, float]:
        # assume two sensors on the left and right
        # left sensor position
        lsx = self.x + self.r_robot * cos(self.a+self.b)
        lsy = self.y + self.r_robot * sin(self.a+self.b)
        lsa = self.a + self.b

        # right sensor position
        rsx = self.x + self.r_robot * cos(self.a-self.b)
        rsy = self.y + self.r_robot * sin(self.a-self.b)
        rsa = self.a - self.b

        return (lsx, lsy, lsa, rsx, rsy, rsa)
    
    def sensor_value_at(self, sx, sy, sa):
        l = self.light
        # directional sensors
        distance = np.sqrt( (l.x-sx)**2 + (l.y-sy)**2 )

        sv = ((l.x-sx)/distance, (l.y-sy)/distance,) # normalised vector (tuple) from sensor position to light position
        # sensor direction vector
        sd = (cos(sa), sin(sa),) # normalised vector

        # clamping the dot product
        dot_prod = sv[0] * sd[0] + sv[1] * sd[1]
        dot_prod = dot_prod if dot_prod > 0 else 0 # no negative sensor values

        s = l.intensity * ( 1.0 / (1.0 + distance) ) * dot_prod

        return s

    def prep(self):
        lsx, lsy, lsa, rsx, rsy, rsa = self.get_sensor_positions()

        # sensor excitation values
        l_s = self.sensor_value_at(lsx, lsy, lsa)
        r_s = self.sensor_value_at(rsx, rsy, rsa)

        # genes = [ls_lm, ls_rm, rs_lm, rs_rm]
        l_m = self.genes.genes[0] * l_s + self.genes.genes[2] * r_s  # note that there is no gene for starting velocity
        r_m = self.genes.genes[1] * l_s + self.genes.genes[3] * r_s

        # how x, y, and a change from the two motors, m_l and m_r
        # save it as part of the vehicle object
        self.dxdt = cos(self.a)*(l_m + r_m) * self.r_wheel
        self.dydt = sin(self.a)*(l_m + r_m) *  self.r_wheel
        self.dadt = self.r_wheel * (r_m - l_m) / (2*self.r_robot)

        return l_s, r_s, l_m, r_m

    def update(self, DT) -> None:
        """
        Updates the change in state using Euler integration
        """
        self.x += self.dxdt * DT 
        self.y += self.dydt * DT 
        self.a += self.dadt * DT 
    

