import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt


def calc_total_time(path, time_matrix):
    # bereken de cumulatieve tijd van alle stappen in een traject (oplossing)
    time_total = time_matrix.iloc[0, path[0]]               # tijd van start naar 1e punt (begin stap)
    for i in range(len(path) - 1):                          # itereren over alle stappen
        a = path[i]                                         # beginpunt stap
        b = path[i + 1]                                     # eindpunt stap
        time_step = time_matrix.iloc[(20 * i) + a, (20 * (i + 1)) + b]  # tijd waarde voor stap uit tijd matrix
        time_total = time_total + time_step                 # voeg staptijd toe aan cumulatieve tijd
    time_total = time_total + time_matrix.iloc[(180 + path[-1]), 201]   # tijdstap van laatste punt naar eindpunt
    return(time_total)                                      # output: totale tijd van gehele traject

def calc_p_surv(path, surv_matrix):
    # bereken de overlevingskans van de UAV over het traject(oplossing)
    # waarde in matrix staan gegeven als detectie/kill kansen (?)
    p_surv = 1 - surv_matrix.iloc[0, path[0]]           # ovelevingskans start naar 1e punt
    for i in range(len(path) - 1):                      # itereren over alle stappen
        a = path[i]                                     # beginpunt stap
        b = path[i + 1]                                 # eindpunt stap
        p_surv_step = surv_matrix.iloc[(20 * i) + a, (20 * (i + 1)) + b]    # detectie/kill kans voor stap uit matrix
        p_surv = p_surv * (1 - p_surv_step)                 # updaten cumulatieve overlevingskans
    p_surv = p_surv * (1 - surv_matrix.iloc[(180 + path[-1]), 201])     # updaten met overlevingskans naar eindpunt
    return(p_surv)                                          # output: overlevingskans van traject (oplossing)

def choose_step(ph_array):
    """bepaal welke stap de mier gaat maken aan de hand van feromonen niveaus per stap"""
    step_sum = np.sum(ph_array)
    step_chance = []
    for i in range(ph_array.size):
        step_chance.append(ph_array[i] / step_sum)
    k = np.random.random_sample()
    f = 0
    for i in range(ph_array.size):
        f = f + step_chance[i]
        if k <= f:
            keuze = i
            break
    return(keuze+1)

def create_path(ph_matrix):
    path = [0]
    for i in range(10):  # 11 stappen
        ph_array = ph_matrix.iloc[(20 * i) + path[i]]
        ph_array = ph_array[(20 * i) + 1: (20 * (i + 1)) + 1]
        ph_array = ph_array.to_numpy()
        keuze = choose_step(ph_array)
        path.append(keuze)
    del path[0]
    return(path)

def apply_pheromones(ph_matrix, path, pheromone):
    path.append(0)
    ph_matrix.iloc[0, path[0]] = ph_matrix.iloc[0, path[0]] + pheromone
    for i in range(10 - 1):
        a = path[i]         # beginpunt stap
        b = path[i + 1]
        ph_matrix.iloc[(20 * i) + a, (20 * (i + 1)) + b] = ph_matrix.iloc[(20 * i) + a, (20 * (i + 1)) + b] + pheromone
    ph_matrix.iloc[(180 + path[-1]), 201] = ph_matrix.iloc[(180 + path[-1]), 201] + pheromone
    return(ph_matrix)

def plot_path(path):
    path.append(10)
    path.insert(0,10)
    plt.plot(path)
    plt.ylabel('point')
    plt.xlabel('step')
    plt.xticks(range(12))
    plt.yticks(range(0, 22))
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    font = {'family': 'serif',
            'color': 'red',
            'weight': 'normal',
            'size': 16,
            }
    plt.text(-0.5, 10, r'A', fontdict=font)
    plt.text(11, 10, r'B', fontdict=font)
    plt.show()