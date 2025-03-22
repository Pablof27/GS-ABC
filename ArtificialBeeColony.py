import numpy as np
import random
from dataclasses import dataclass
from EventList import EventList

@dataclass
class Parameters:
    N: int
    limit: int
    max_trials: int

class ArtificialBeeColony:
    
    def __init__(self, psmodel):
        self.psmodel = psmodel
    
    def optimize(self, params: Parameters):
        trials = 0
        food_sources = [EventList(self.psmodel) for _ in range(params.N)]
        updates = np.zeros(params.N, dtype=np.int32)
        best_solution = min(food_sources, key=lambda s: s.get_makespan())
        while trials < params.max_trials:
            
            for i in range(params.N):
                updates[i] += 1
                new_solution = food_sources[i].generate_new_local_solution()
                if new_solution.get_makespan() < food_sources[i].get_makespan():
                    food_sources[i] = new_solution
                    updates[i] = 0
                    
            for i in range(params.N):
                random_indexes = random.sample(range(params.N), 3)
                selected = min(random_indexes, key=lambda x: food_sources[x].get_makespan())
                updates[selected] += 1
                new_solution = food_sources[selected].generate_new_local_solution()
                if new_solution.get_makespan() < food_sources[selected].get_makespan():
                    food_sources[selected] = new_solution
                    updates[selected] = 0
                    
            iteration_best = min(food_sources, key=lambda s: s.get_makespan())
            if iteration_best.get_makespan() < best_solution.get_makespan():
                best_solution = iteration_best
                trials = 0
            else:
                trials += 1
                
            selected = np.where(updates > params.limit)[0]
            for i in selected:
                food_sources[i] = EventList(self.psmodel)
                updates[i] = 0
        
        return best_solution