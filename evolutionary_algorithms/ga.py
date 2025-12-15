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
            sm_h[i].append(v.prep())
        
        # update x, y, a and save them in histories
        for i in range(n):
            v = population[i]
            v.update(DT)
            x, y, a = v.get_state()
            x_h[i].append(x); y_h[i].append(y); a_h[i].append(a)
    
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

    


if __name__ == '__main__':
    initial_population = make_population(1, 3)
    print("init genes", [v.genes.genes for v in initial_population])


    population = copy.deepcopy(initial_population)
    scores = evaluate(initial_population, population, 3)

    for i in range(1000):
        # print(i)
        mutate(population, 0.5, 0.1)
        epoch(population, 10, 0.1)
        # print(population)
        if i == 0 or i == 999:
            print(evaluate(initial_population, population, 3))
            print("after genes", [v.genes.genes for v in population])



