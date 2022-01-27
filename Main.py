import numpy as np
import pandas as pd
import math
import random
import functions as fnc
import classes
import matplotlib.pyplot as plt
import time





# Import datafiles:
surv_matrix = pd.read_csv('S.txt', header=None)
time_matrix = pd.read_csv('T.txt', header=None)

# control values
time_max = 24       # maximum length of path
ph_begin = 1        # base level of pheromones
num_iter = 100    # number of iterations
n_ants = 100       # number of ants per iteration
rho = 0.9          # verdampingsfactor per iteratie
ph_booster = 5      # feromoon booster bij nieuwe beste oplossing


start_time = time.time()
# random initial solution
# np.random.seed(100)
init_sol = np.random.randint(1, 21, size=10)
# evaluate initial solution, set as initial best solution
time_best = fnc.calc_total_time(init_sol, time_matrix)
p_surv_best = fnc.calc_p_surv(init_sol, surv_matrix)
path_best = init_sol


# create initial pheromone matrix
ph_matrix = np.full((202,202), ph_begin)  # vullen met feromonen begin waarde 1
ph_matrix = pd.DataFrame(ph_matrix, index=None, columns=None)



# start ACO
solutions = [p_surv_best]
p_surv_best_sim = []
p_surv_avg_sim = []
for i in range(num_iter):
    print(i)
    # np.random.seed(i)
    # create group of ants, create path, evaluate path & pheromone
    ants = {}
    for i in range(n_ants):
        ants[i] = classes.Ant()
        # create path
        ants[i].create_path(ph_matrix, surv_matrix)
        # evaluate path
        ants[i].calc_total_time(time_matrix)
        ants[i].calc_p_surv(surv_matrix)
        # calculate pheromones
        ants[i].calc_pheromone(p_surv_best)
        # better path bonus
        if ants[i].p_surv >= p_surv_best:
            ants[i].pheromone = ants[i].pheromone * ph_booster

    # evaporate pheromonen
    ph_matrix = ph_matrix.multiply(rho)

    # apply pheromones from paths only if path <max_length
    for i in range(len(ants)):
        if ants[i].time <= time_max:
            ph_matrix = ants[i].apply_pheromone(ph_matrix)


    # check if a path beats the best, update best solution
    for i in range(len(ants)):
        if ants[i].time <= time_max:
            if ants[i].p_surv > p_surv_best:
                p_surv_best = ants[i].p_surv
                path_best = ants[i].path
                time_best = ants[i].time
 #               fnc.plot_path(path_best)
    solutions.append(p_surv_best)

    # collection and presentation of metaheuristic data
    ants_psurv = []
    for i in range(len(ants)):
        if ants[i].time <= time_max:
            ants_psurv.append(ants[i].p_surv)
    p_surv_best_sim.append(max(ants_psurv))
    p_surv_avg_sim.append(sum(ants_psurv) / len(ants_psurv))


# write ph_matrix to file
ph_matrix = round(ph_matrix, 1)
ph_matrix.to_csv('ph_matrix.txt', index=False, header=False)

print(path_best)
print(p_surv_best)


fnc.plot_path(path_best)
plt.plot(solutions)
plt.ylabel('best solutions')
plt.xlabel('iteration')
plt.show()

plt.plot(p_surv_best_sim, label='best of population')
plt.plot(p_surv_avg_sim, label = 'average of population')
plt.legend(loc="lower right")
plt.ylabel('Chance of survival')
plt.xlabel('Iteration')
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))