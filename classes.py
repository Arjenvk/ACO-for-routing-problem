import numpy as np
import pandas as pd
import math
import random


class Ant:
    def __init__(self):
        self.path = []
        self.time = 0
        self.p_surv = 0
        self.pheromone = 0

    def choose_step(self, choice_array):
        """bepaal welke stap de mier gaat maken aan de hand van feromonen niveaus per stap"""
        step_sum = np.sum(choice_array)
        step_chance = []
        for i in range(choice_array.size):
            step_chance.append(choice_array[i] / step_sum)
        k = np.random.random_sample()
        f = 0
        for i in range(choice_array.size):
            f = f + step_chance[i]
            if k <= f:
                keuze = i
                break
        return(keuze + 1)

    def create_path(self, ph_matrix, surv_matrix):
        ph_array = ph_matrix.iloc[0]
        ph_array = ph_array[1:21]
        ph_array = ph_array.to_numpy()
        surv_array = surv_matrix.iloc[0]
        surv_array = surv_array[1:21]
        surv_array = surv_array.to_numpy()
        choice_array = np.multiply(ph_array, (1-surv_array))
        keuze = self.choose_step(choice_array)
        self.path.append(keuze)
        for i in range(9):  # 9 stappen
            ph_array = ph_matrix.iloc[(20 * i) + self.path[i]]
            ph_array = ph_array[(20 * (i+1) + 1): (20 * (i + 2)) + 1]
            ph_array = ph_array.to_numpy()
            surv_array = surv_matrix.iloc[(20 * i) + self.path[i]]
            surv_array = surv_array[(20 * (i+1) + 1): (20 * (i + 2)) + 1]
            surv_array = surv_array.to_numpy()
            choice_array = np.multiply(ph_array, (1-surv_array))
            keuze = self.choose_step(choice_array)
            self.path.append(keuze)

    def calc_total_time(self, time_matrix):
        # bereken de cumulatieve tijd van alle stappen in een traject (oplossing)
        time_total = time_matrix.iloc[0, self.path[0]]               # tijd van start naar 1e punt (begin stap)
        for i in range(9):                                           # itereren over eerste 9 stappen
            a = self.path[i]                                         # beginpunt stap
            b = self.path[i + 1]                                     # eindpunt stap
            time_step = time_matrix.iloc[(20 * i) + a, (20 * (i + 1)) + b]  # tijd waarde voor stap uit tijd matrix
            time_total = time_total + time_step                 # voeg staptijd toe aan cumulatieve tijd
        time_total = time_total + time_matrix.iloc[(180 + self.path[-1]), 201]   # tijdstap van laatste punt naar eindpunt
        self.time = time_total                                  # output: totale tijd van gehele traject

    def calc_p_surv(self, surv_matrix):
        # bereken de overlevingskans van de UAV over het traject(oplossing)
        # waarde in matrix staan gegeven als detectie/kill kansen (?)
        p_surv = 1 - surv_matrix.iloc[0, self.path[0]]           # ovelevingskans start naar 1e punt
        for i in range(9):                                       # itereren over eerste 9 stappen
            a = self.path[i]                                     # beginpunt stap
            b = self.path[i + 1]                                 # eindpunt stap
            p_surv_step = surv_matrix.iloc[(20 * i) + a, (20 * (i + 1)) + b]    # detectie/kill kans voor stap uit matrix
            p_surv = p_surv * (1 - p_surv_step)                 # updaten cumulatieve overlevingskans
        p_surv = p_surv * (1 - surv_matrix.iloc[(180 + self.path[-1]), 201])     # updaten met overlevingskans naar eindpunt
        self.p_surv = p_surv

    def calc_pheromone(self, p_surv_best):
        self.pheromone = self.p_surv / p_surv_best


    def apply_pheromone(self, ph_matrix):
        ph_matrix.iloc[0, self.path[0]] = ph_matrix.iloc[0, self.path[0]] + self.pheromone
        for i in range(10 - 1):
            a = self.path[i]  # beginpunt stap
            b = self.path[i + 1]
            ph_matrix.iloc[(20 * i) + a, (20 * (i + 1)) + b] = ph_matrix.iloc[(20 * i) + a, (20 * (i + 1)) + b] + self.pheromone
        ph_matrix.iloc[(180 + self.path[-1]), 201] = ph_matrix.iloc[(180 + self.path[-1]), 201] + self.pheromone
        return(ph_matrix)





