import numpy as np
import random
from dataclasses import dataclass
from EventList2 import EventList
from Utils import positional_entropy, topological_sort, calculate_unique_individuals
import copy

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

        final_population = [EventList(psmodel=self.psmodel, jobs=p) for p in population] + [EventList(psmodel=self.psmodel) for _ in range(int(N/2))]
        return final_population[:N]

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
        
    def employeed_phase(self):
        self.updates += 1
        num_sources = len(self.food_sources)
        step_choices = [1, 2, 3]
        step_weights = [0.9, 0.07, 0.03]
        random_steps = np.random.choice(step_choices, size=num_sources, p=step_weights)
        
        for i, (current_solution, steps) in enumerate(zip(self.food_sources, random_steps)):
            new_solution = current_solution.generate_new_local_solution(steps=steps)

            if new_solution.get_makespan() < current_solution.get_makespan():
                self.food_sources[i] = new_solution
                self.updates[i] = 0
                    
    def onlooker_phase(self):
        num_sources = len(self.food_sources)
        all_makespans = np.array([fs.get_makespan() for fs in self.food_sources])
        tournament_indices = np.random.randint(0, num_sources, size=(num_sources, 3))
        tournament_makespans = all_makespans[tournament_indices]
        winner_local_indices = np.argmin(tournament_makespans, axis=1)
        selected_indices = tournament_indices[np.arange(num_sources), winner_local_indices]
        for selected in selected_indices:
            self.updates[selected] += 1
            new_solution = self.food_sources[selected].generate_new_local_solution(steps=1)
            if new_solution.get_makespan() < self.food_sources[selected].get_makespan():
                self.food_sources[selected] = new_solution
                self.updates[selected] = 0

    def probability(self, trials, max_limit):
        return 1/(1.1 + (trials * 3/max_limit)**3)
    
    def optimize_old(self, params: Parameters = None, mode="abc", init="random"):
        
        self.params = self.params if params is None else params
        p = self.params
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
            
            self.employeed_phase()
            self.onlooker_phase()

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
            if mode == "abc":
                for i in selected:
                    self.food_sources[i] = EventList(psmodel=self.psmodel)
            elif mode.startswith("gs-abc"):
                for i in selected:
                    if random.random() < self.probability(trials, p.max_trials):
                        self.food_sources[i] = EventList(psmodel=self.psmodel)
                        continue
                    self.food_sources[i] = iteration_best.recombine_solution(self.food_sources[i]) if random.random() < 0.5 else self.food_sources[i].recombine_solution(iteration_best)
                    if random.random() < params.mr:
                        self.food_sources[i] = self.food_sources[i].swap_new_solution(iterations=random.choices([1, 2, 3], weights=[0.75, 0.15, 0.1], k=1)[0])
                    self.updates[i] = 0
            else:
                raise ValueError(f'{mode} is not a valid mode')

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

        
    def optimize(self, params: Parameters = None, mode="abc", init="random"):
        self.params = self.params if params is None else params
        p = self.params
        
        # --- Initialization ---
        if init == "random":
            self.food_sources = [EventList(self.psmodel) for _ in range(p.N)]
        elif init == "mcmc":
            self.food_sources = self.init_population(p.N)
        else:
            raise ValueError(f'{init} not a valid initialization')
        
        self.updates = np.zeros(p.N, dtype=np.int32)
        
        all_makespans = np.array([s.get_makespan() for s in self.food_sources])
        best_idx = np.argmin(all_makespans)
        best_solution = copy.deepcopy(self.food_sources[best_idx])
        
        history = [best_solution]
        permutations = [[job.id for job in solution.jobs] for solution in self.food_sources]
        population_diversity = [positional_entropy(permutations)[1]]
        nscout_bees = [0]
        nunique_individuals = [calculate_unique_individuals(self.food_sources)]
        
        trials = 0
        while trials < p.max_trials and best_solution.get_makespan() > self.psmodel.best_known and len(history) < 10*len(best_solution.jobs[2:])*4:
            
            # if (len(history) - 1) % 100 == 0:
            #     print(f"")
            # --- Main Algorithm Phases ---
            # NOTE: These phases modify `self.food_sources` in place
            self.employeed_phase()
            self.onlooker_phase()

            # --- Update and Evaluation (Optimized) ---
            all_makespans = np.array([s.get_makespan() for s in self.food_sources])
            iter_best_idx = np.argmin(all_makespans)
            
            # Local search logic (optional)
            if p.l != -1 and (trials - 1) % p.l == 0:
                self.food_sources[iter_best_idx].shift_foward_search()
                all_makespans[iter_best_idx] = self.food_sources[iter_best_idx].get_makespan()

            # Check for new global best
            if all_makespans[iter_best_idx] < best_solution.get_makespan():
                best_solution = copy.deepcopy(self.food_sources[iter_best_idx])
                if p.l != -1:
                    best_solution.shift_foward_search()
                trials = 0
            else:
                trials += 1
            
            history.append(best_solution)

            # --- Scout Bee Phase (Optimized) ---
            scout_indices = np.where(self.updates > p.limit)[0]
            if scout_indices.size > 0:
                if mode == "abc":
                    for i in scout_indices:
                        self.food_sources[i] = EventList(psmodel=self.psmodel)
                elif mode.startswith("gs-abc"):
                    # Vectorize random choices for gs-abc mode
                    prob_roll = np.random.random(len(scout_indices))
                    recombine_roll = np.random.random(len(scout_indices))
                    mutation_roll = np.random.random(len(scout_indices))

                    for j, i in enumerate(scout_indices):
                        if prob_roll[j] < self.probability(trials, p.max_trials):
                            self.food_sources[i] = EventList(psmodel=self.psmodel)
                            continue
                        
                        iter_best_solution = self.food_sources[iter_best_idx]
                        if recombine_roll[j] < 0.5:
                            self.food_sources[i] = iter_best_solution.recombine_solution(self.food_sources[i])
                        else:
                            self.food_sources[i] = self.food_sources[i].recombine_solution(iter_best_solution)

                        if mutation_roll[j] < p.mr:
                            self.food_sources[i] = self.food_sources[i].swap_new_solution(iterations=random.choices([1, 2, 3], weights=[0.75, 0.15, 0.1], k=1)[0])
                else:
                    raise ValueError(f'{mode} is not a valid mode')

                # Reset updates for scouted bees
                self.updates[scout_indices] = 0
            
            # (History tracking for diversity etc. would go here)
            permutations = [[job.id for job in solution.jobs] for solution in self.food_sources]
            population_diversity.append(positional_entropy(permutations)[1])
            nunique_individuals.append(calculate_unique_individuals(self.food_sources))

        self.population_divsersity = population_diversity
        self.nscout_bees = nscout_bees
        self.nunique_individuals = nunique_individuals
        self.history = history
        return self.history[-1]