import numpy as np
import random
from dataclasses import dataclass
from EventList import EventList
from typing import Tuple
import multiprocessing as mp

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
                
    def employee_and_onlooker_phase(self, slice: Tuple[int, int]):
        self.employeed_phase(slice)
        self.onlooker_phase(slice)
    
    def optimize(self, params: Parameters = None, num_workers: int = 1):
        
        p = params if params else self.params
        slices = [(i*p.N//num_workers, (i+1)*p.N//num_workers) for i in range(num_workers)]

        trials = 0
        self.food_sources = [EventList(self.psmodel) for _ in range(p.N)]
        self.updates = np.zeros(p.N, dtype=np.int32)
        best_solution = min(self.food_sources, key=lambda s: s.get_makespan())
        
        with mp.Pool(num_workers) as pool:
            while trials < p.max_trials:
                
                pool.map(self.employeed_phase, slices)
                        
                iteration_best = min(self.food_sources, key=lambda s: s.get_makespan())
                if iteration_best.get_makespan() < best_solution.get_makespan():
                    best_solution = iteration_best
                    trials = 0
                else:
                    trials += 1
                    
                selected = np.where(self.updates > p.limit)[0]
                for i in selected:
                    self.food_sources[i] = EventList(self.psmodel)
                    self.updates[i] = 0
        
        return best_solution