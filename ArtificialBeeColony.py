import numpy as np
import random
from dataclasses import dataclass
from EventList import EventList
from typing import Tuple
import multiprocessing as mp
from Utils import positional_entropy, average_pairwise_distance

@dataclass
class Parameters:
    N: int
    limit: int
    max_trials: int

class ArtificialBeeColony:
    
    def __init__(self, psmodel):
        self.psmodel = psmodel
        
    def set_parameters(self, N, limit, max_trials):
        self.params = Parameters(N, limit, max_trials)
        
    def employeed_phase(self, slice: Tuple[int, int]):
        beg, end = slice
        for i in range(beg, end):
            self.updates[i] += 1
            new_solution = self.food_sources[i].generate_new_local_solution()
            if new_solution.get_makespan() < self.food_sources[i].get_makespan():
                self.food_sources[i] = new_solution
                self.updates[i] = 0
                    
    def onlooker_phase(self, slice: Tuple[int, int]):
        beg, end = slice
        for i in range(beg, end):
            random_indexes = random.sample(range(beg, end), 3)
            selected = min(random_indexes, key=lambda x: self.food_sources[x].get_makespan())
            self.updates[selected] += 1
            new_solution = self.food_sources[selected].generate_new_local_solution()
            if new_solution.get_makespan() < self.food_sources[selected].get_makespan():
                self.food_sources[selected] = new_solution
                self.updates[selected] = 0
                
    
    def optimize(self, params: Parameters = None, num_workers: int = 1):
        
        p = params if params else self.params
        slices = [(i*p.N//num_workers, (i+1)*p.N//num_workers) for i in range(num_workers)]

        trials = 0
        self.food_sources = [EventList(self.psmodel) for _ in range(p.N)]
        self.updates = np.zeros(p.N, dtype=np.int32)
        best_solution = min(self.food_sources, key=lambda s: s.get_makespan())
        best_evolution = [best_solution]

        permutations = [[job.id for job in solution.jobs] for solution in self.food_sources]
        population_diversity = [positional_entropy(permutations)[1]]
        average_distance = [average_pairwise_distance(permutations)]
        
        while trials < p.max_trials:
            
            self.employeed_phase(slices[0])
            self.onlooker_phase(slices[0])
                    
            iteration_best = min(self.food_sources, key=lambda s: s.get_makespan())
            if iteration_best.get_makespan() < best_solution.get_makespan():
                best_solution = iteration_best
                trials = 0
            else:
                trials += 1
            best_evolution.append(best_solution)
                
            selected = np.where(self.updates > p.limit)[0]
            for i in selected:
                self.food_sources[i] = best_solution.recombine_solution(self.food_sources[i]) if random.random() < 0.5 else self.food_sources[i].recombine_solution(best_solution)
                self.updates[i] = 0

            permutations = [[job.id for job in solution.jobs] for solution in self.food_sources]
            population_diversity.append(positional_entropy(permutations)[1])
            average_distance.append(average_pairwise_distance(permutations))
        
        self.population_divsersity = population_diversity
        self.average_distance = average_distance
        
        return best_evolution