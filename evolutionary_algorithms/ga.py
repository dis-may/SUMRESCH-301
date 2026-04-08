from pylab import *
from evolvable_vehicles import Vehicle, Genotype, Light
import copy

def make_population(n, size) -> list[Vehicle]:
    """
    n: int -> the number of vehicles
    size: int -> vehicles will have x, y randomly chosen from [-size, size]
    """
    vehicles = []
    for _ in range(n):
        x = np.random.uniform(-size, size)
        y = np.random.uniform(-size, size)
        a = np.random.uniform(-size, size)
        gene = Genotype(0, 0, 0, 0) # placeholders for genes
        gene.randomise()
        v = Vehicle(gene, x, y, a)
        vehicles.append(v)
    
    return vehicles


def epoch(population: list[Vehicle], duration, DT) -> list[list[float], list[float], list[float], list[int], list[tuple[float, float, float, float]]]:
    """
    Use Euler integration to run a simulation for 'duration' over time steps size DT
    Returns the histories of the x, y, a, t, and SM states
    Changes in the population are made in place
    """
    # n_its = duration // DT

    n = len(population)

    x_h = [list() for _ in range(n)]
    y_h = [list() for _ in range(n)]
    a_h = [list() for _ in range(n)]
    t_h = []

    sm_h = [list() for _ in range(n)]

    # initialising time
    t_h.append(0)

    # initialise the arrays with the initial position of the vehicles
    for i in range(n):
        v = population[i]
        x, y, a = v.get_state()
        x_h[i].append(x); y_h[i].append(y); a_h[i].append(a)

    # iterate through the second time step onwards
    for t in np.arange(DT, duration+DT, DT):
        # STEP 
        t_h.append(t)

        # calculate dx, dy, da (saved in vehicle object)
        for i in range(n):
            # save SM history (s_l, s_r, m_l, m_r)
            v = population[i]
            l_s, r_s, l_m,  r_m = v.prep()
            sm_h[i].append((l_s, r_s, l_m,  r_m,))
        
        # update x, y, a and save them in histories
        for i in range(n):
            v = population[i]
            v.update(DT)
            x, y, a = v.get_state()
            x_h[i].append(x); y_h[i].append(y); a_h[i].append(a)
            # print(x_h, y_h, a_h)
    
    # save the last SM state experienced
    for i in range(n):
        sm_h[i].append(v.prep())
        
    
    return (x_h, y_h, a_h, t_h, sm_h)


def evaluate(initial_positions: list[Vehicle],
             population: list[Vehicle], 
             size: int, # to calculate the maximum distance between vehicle and light, same as in def make_population
             light_source=Light(0,0)) -> list[float]:
    """
    Evaluate the fitness of each individual in the population
    Score is based on the distance to the light source, ranging from [-np.sqrt(size**2 + size**2) 
    could also do travelling towards the light? evaluate if final - initial is closer, as opposed to just final position
    """
    l = light_source
    scores = []
    for i in range(len(population)):
        vi = initial_positions[i] # initial
        vf = population[i] # final
        # initial_distance = np.sqrt( (l.x-vi.x)**2 + (l.y-vi.y)**2 )
        final_distance = np.sqrt( (l.x-vf.x)**2 + (l.y-vf.y)**2 )

        max_initial_distance = np.sqrt(size**2 + size**2)
        score = (1 - (final_distance + 1E-16 / max_initial_distance))
        if score < 0:
            score = 0 # it is possible that final distance > max_initial_distance, so we clamp
        scores.append(score)
    
    return scores


def mutate(population: list[Vehicle], mutation_rate, mutation_strength):
    """
    Go through each gene and have a chance of mutating each one
    """
    for v in population:
        # for each gene in each vehicle genotype
        # print("genotype before", v.genes.genes)
        for i in range(v.genes.n):
            if np.random.uniform(0, 1) < mutation_rate:
                v.genes.genes[i] += np.random.normal(0, mutation_strength)
        
        # print("genotype after", v.genes.genes)

def display(title, x_h, y_h, a_h, t_h, x_lim, y_lim):
    
    plt.rcParams.update({'font.size': 10})
    fig_1 = plt.figure(figsize=(4,4))

    axes_1 = fig_1.add_axes([0.1, 0.1, 0.8, 0.8])

    # plotting all vehicles in the simulation
    for i in range(len(x_h)):
        # plot(x_h[i][0],y_h[i][0],'bo')
        axes_1.plot(x_h[i], y_h[i])
        axes_1.plot(x_h[i][-1],y_h[i][-1],'ro', ms=3)
        # axes_1.plot(x_h[i][0],y_h[i][0],'rx')

        # # plot arrow for final position
        # speed = self.vehicles[i].m_l + self.vehicles[i].m_r


    # plotting all lights in the simulation
    # for l in self.lights:
    #     axes_1.plot(l.x, l.y,'bx') 
    
    ## fix up the figure to look decent    
    axes_1.set_aspect('equal', adjustable='box')
    axes_1.set_xlim(-x_lim,x_lim); axes_1.set_ylim(-y_lim,y_lim)
    axes_1.set_xticks(np.arange(-x_lim, x_lim+5, 5))
    axes_1.set_yticks(np.arange(-y_lim, y_lim+5, 5))
    # axes_1 = plt.gca()
    # axes_1.set_aspect('equal', adjustable='box')
    axes_1.grid()
    # axes_1.set_xlabel('x'); axes_1.set_ylabel('y')
    axes_1.set_title(title)
    
    # savefig(title + "_random_larger_font", dpi=300)
    show()
    

def testing_a():
    # g = Genotype(0, 1, 1, 0) ## aggression
    g = Genotype(0.6641450727677876, 0.229184857943735, 0.5261044852050254, 0.3767926197637974)
    # g = Genotype(0, 1, 0, 1) ## fear

    v = Vehicle(g, -2, -2, np.pi/4)
    x_h, y_h, a_h, t_h, sm_h = epoch([v], 5, 0.1)
    display("shit", x_h, y_h, a_h, t_h,  4, 4)

def old_testing():
    n = 1
    size = 3 # -size and +size is the box where vehicles spawn
    initial_population = make_population(n, size)
    print("init genes", [v.genes.genes for v in initial_population])


    population = copy.deepcopy(initial_population)
    scores = evaluate(initial_population, population, size)

    for i in range(1000):
        # print(i)
        mutate(population, 0.5, 0.1)
        epoch(population, 10, 0.1)
        # print(population)
        if i == 0 or i == 999:
            print(evaluate(initial_population, population, size))
            print("after genes", [v.genes.genes for v in population])
    
    x_h, y_h, a_h, t_h, sm_h = epoch(population, 10, 0.1)
    # for x in x_h:
    #     for stuff in x:
    #         print(stuff, end=" ")
    display("Test", x_h, y_h, a_h, t_h, 5, 5)

    


if __name__ == '__main__':
    testing_a()



