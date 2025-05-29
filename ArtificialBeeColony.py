import numpy as np
import random
from dataclasses import dataclass
from EventList import EventList
from typing import Tuple
import multiprocessing as mp
from Utils import positional_entropy, topological_sort, calculate_unique_individuals

@dataclass
class Parameters:
    N: int
    limit: int
    max_trials: int
    mr: float = 0.1
    l: int = -1

class ArtificialBeeColony:
    
    def __init__(self, psmodel):
        self.psmodel = psmodel

    def init_population(self, N):
        ldf = topological_sort(self.psmodel.jobs, metric='ldf')
        sdf = topological_sort(self.psmodel.jobs, metric='sdf')
        mrf = topological_sort(self.psmodel.jobs, metric='mrf')
        lrf = topological_sort(self.psmodel.jobs, metric='lrf')
        rnd = topological_sort(self.psmodel.jobs, metric='random')

        population = self.mcmc_sampling(ldf, N/8, 0, 25)
        population += self.mcmc_sampling(sdf, N/8, 0, 25)
        population += self.mcmc_sampling(mrf, N/8, 0, 25)
        population += self.mcmc_sampling(lrf, N/8, 0, 25)

        return [EventList(psmodel=self.psmodel, jobs=p) for p in population] + [EventList(psmodel=self.psmodel) for _ in range(int(N/2))]

    def mcmc_sampling(self, permutation, num_sample, burn_in: int, spacing: int):
        samples = []
        iterations = burn_in + num_sample * spacing

        for t in range(int(iterations)):
            idx = random.randint(1, len(self.psmodel.jobs)-3)
            if EventList.can_swap(permutation, idx):
                permutation[idx], permutation[idx+1] = permutation[idx+1], permutation[idx]
            if t >= burn_in and (t-burn_in)%spacing == 0:
                samples.append(permutation[:]) 

        return samples
    
    def set_parameters(self, N, limit, max_trials):
        self.params = Parameters(N, limit, max_trials)
        
    def employeed_phase(self, slice: Tuple[int, int]):
        beg, end = slice
        for i in range(beg, end):
            self.updates[i] += 1
            new_solution = self.food_sources[i].generate_new_local_solution(steps=random.choices([1, 2, 3], weights=[0.9, 0.07, 0.03], k=1)[0])
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
                
    
    def optimize(self, params: Parameters = None, num_workers: int = 1, mode="basic", init="random"):
        
        self.params = self.params if params is None else params
        p = self.params
        slices = [(i*p.N//num_workers, (i+1)*p.N//num_workers) for i in range(num_workers)]

        trials = 0
        if init == "random":
            self.food_sources = [EventList(self.psmodel) for _ in range(p.N)]
        elif init == "mcmc":
            self.food_sources = self.init_population(p.N)
        else:
            raise ValueError(f'{init} not a valid initialization')
        self.updates = np.zeros(p.N, dtype=np.int32)
        best_solution = min(self.food_sources, key=lambda s: s.get_makespan())
        best_evolution = [best_solution]

        permutations = [[job.id for job in solution.jobs] for solution in self.food_sources]
        population_diversity = [positional_entropy(permutations)[1]]
        nscout_bees = [0]
        nunique_individuals = [calculate_unique_individuals(self.food_sources)]
        # average_distance = [average_pairwise_distance(permutations)]
        # k_average_distance = [average_pairwise_distance(permutations, metric='kendall')]
        
        while trials < p.max_trials and best_solution.get_makespan() > self.psmodel.best_known:
            
            self.employeed_phase(slices[0])
            self.onlooker_phase(slices[0])

            iteration_best = min(self.food_sources, key=lambda s: s.get_makespan())
            if params.l != -1 and trials % params.l == 0 and trials != 0:
                iteration_best.shift_foward_search()
            if iteration_best.get_makespan() < best_solution.get_makespan():
                best_solution = iteration_best
                if params.l != -1:
                    best_solution.shift_foward_search()
                trials = 0
            else:
                trials += 1
            best_evolution.append(best_solution)
                
            selected = np.where(self.updates > p.limit)[0]
            nscout_bees.append(len(selected))
            for i in selected:
                if mode == "basic":
                    self.food_sources[i] = EventList(psmodel=self.psmodel)
                elif mode == "variant":
                    self.food_sources[i] = best_solution.recombine_solution(EventList(psmodel=self.psmodel)) if random.random() < 0.5 else EventList(psmodel=self.psmodel).recombine_solution(best_solution)
                else:
                    raise ValueError(f'{mode} is not a valid mode')
                r = random.random()
                if r < params.mr:
                    self.food_sources[i] = self.food_sources[i].swap_new_solution(iterations=random.choices([1, 2, 3], weights=[0.75, 0.15, 0.1], k=1)[0])
                self.updates[i] = 0

            permutations = [[job.id for job in solution.jobs] for solution in self.food_sources]
            population_diversity.append(positional_entropy(permutations)[1])
            nunique_individuals.append(calculate_unique_individuals(self.food_sources))
            # average_distance.append(average_pairwise_distance(permutations))
            # k_average_distance.append(average_pairwise_distance(permutations, metric='kendall'))
        
        self.population_divsersity = population_diversity
        self.nscout_bees = nscout_bees
        self.history = best_evolution
        self.nunique_individuals = nunique_individuals
        # self.average_distance = average_distance
        # self.k_average_distance = k_average_distance
        
        return best_evolution[-1]