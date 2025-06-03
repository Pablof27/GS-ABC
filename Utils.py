from typing import List, Any
import random
from SchedulingProblem import Job, ProjectSchedulingModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import json
from dataclasses import dataclass, asdict, is_dataclass


def topological_sort(jobs: List[Job], metric='random') -> List[Job]:
    in_degree = {job.id: 0 for job in jobs}
    graph = {job.id: [] for job in jobs}
    
    for job in jobs:
        for successor in job.sucessors:
            graph[job.id].append(successor)
            in_degree[successor] += 1
    
    queue = [job for job in jobs if in_degree[job.id] == 0]
    sorted_jobs = []
    
    if metric == 'random':
        select_func = rnd
    elif metric == 'ldf':
        select_func = ldf
    elif metric == 'sdf':
        select_func = sdf
    elif metric == 'mrf':
        select_func = mrf
    elif metric == 'lrf':
        select_func = lrf
    else:
        raise ValueError("Unsupported metric.")
    
    while queue:
        job = queue.pop(select_func(queue))
        sorted_jobs.append(copy.deepcopy(job))
        
        for successor in graph[job.id]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                j = next(job for job in jobs if job.id == successor)
                queue.append(j)
    
    if len(sorted_jobs) != len(jobs):
        raise ValueError("The job graph has at least one cycle.")
    
    return sorted_jobs

def rnd(queue) -> int:
    return random.randint(0, len(queue) - 1)

def ldf(queue) -> int:
    return max(enumerate(queue), key=lambda x: x[1].duration)[0]

def sdf(queue) -> int:
    return min(enumerate(queue), key=lambda x: x[1].duration)[0]

def mrf(queue) -> int:
    return max(enumerate(queue), key=lambda x: np.sum(x[1].resources_needed))[0]

def lrf(queue) -> int:
    return min(enumerate(queue), key=lambda x: np.sum(x[1].resources_needed))[0]

def plot_multiple_schedules(schedules, titles=None, mapping=None):
    num_schedules = len(schedules)
    _, num_times = schedules[0].shape  # All schedules have the same number of columns (time steps)

    # Create a figure with subplots
    fig, axes = plt.subplots(num_schedules, 1, figsize=(20, 2 * num_schedules), squeeze=False)

    # Create a color map for different jobs (shared across all schedules)
    unique_jobs = np.unique(np.concatenate([np.unique(schedule) for schedule in schedules]))
    color_map = plt.get_cmap('viridis', len(unique_jobs))
    
    # supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
    shuffle_map = mapping if mapping else np.random.permutation(len(unique_jobs))

    # Plot each schedule in a separate subplot
    for k, schedule in enumerate(schedules):
        ax = axes[k, 0]
        for i in range(schedule.shape[0]):
            for j in range(num_times):
                job_index = schedule[i, j]
                if job_index != -1:  # Assuming -1 represents idle time
                    color = color_map(shuffle_map[job_index%len(unique_jobs)])
                    ax.add_patch(patches.Rectangle((j, i), 1, 1, color=color))

        # Set labels and ticks for each subplot
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Resource {k+1}')
        if titles is not None and k < len(titles):
            ax.set_title(titles[k])
        ax.set_xticks(np.arange(0, num_times + 1, 1))
        ax.set_yticks(np.arange(0, schedule.shape[0] + 1, 1))
        ax.set_yticklabels([])
        ax.grid(True, which='major', linestyle='-', linewidth=0.25, alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_solution(solution, mapping=None):
    scheme = solution.generate_schedule_scheme()
    plot_multiple_schedules(scheme, mapping=mapping)

import numpy as np
from math import log2

def positional_entropy(permutations):
    """
    Computes average and normalized positional entropy for a list of permutations.

    Args:
        permutations (list of lists): Each inner list is a permutation of integers from 1 to n.

    Returns:
        (float, float): (average_entropy, normalized_entropy)
    """
    if not permutations:
        return 0.0, 0.0

    permutations = np.array(permutations)
    m, n = permutations.shape
    max_entropy = log2(n)
    total_entropy = 0.0

    for i in range(n):
        # Count occurrences of each value at position i
        counts = np.bincount(permutations[:, i], minlength=n + 1)[1:]  # skip index 0
        probs = counts[counts > 0] / m
        entropy = -np.sum(probs * np.log2(probs))
        total_entropy += entropy

    average_entropy = total_entropy / n
    normalized_entropy = average_entropy / max_entropy if max_entropy > 0 else 0.0

    return average_entropy, normalized_entropy

def hamming_distance(p1, p2):
    return sum(a != b for a, b in zip(p1, p2))

def kendall_tau_distance(p1, p2):
    n = len(p1)
    index_map = {val: i for i, val in enumerate(p1)}
    mapped_p2 = [index_map[val] for val in p2]

    # Count inversions in mapped_p2
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if mapped_p2[i] > mapped_p2[j]:
                inversions += 1
    return inversions

def average_pairwise_distance(permutations, metric='hamming'):
    """
    Computes average pairwise distance between permutations using the given metric.
    
    Args:
        permutations (list of lists): Each inner list is a permutation.
        metric (str): 'hamming' or 'kendall'

    Returns:
        float: Average pairwise distance
    """
    m = len(permutations)
    if m < 2:
        return 0.0  # No pairs to compare

    if metric == 'hamming':
        distance_func = hamming_distance
    elif metric == 'kendall':
        distance_func = kendall_tau_distance
    else:
        raise ValueError("Unsupported metric. Use 'hamming' or 'kendall'.")

    total_distance = 0
    count = 0

    for i in range(m - 1):
        for j in range(i + 1, m):
            total_distance += distance_func(permutations[i], permutations[j])
            count += 1

    return total_distance / count

def calculate_unique_individuals(population):
    
    unique_individuals = set()
    
    for individual in population:
        
        jobs_tuples = []
        for job in individual.jobs:
            jobs_tuples.append((job.id, job.start_time))

        jobs_tuples.sort()
        canonical_individual = tuple(jobs_tuples)
        unique_individuals.add(canonical_individual)
    
    return len(unique_individuals)

    
def normalize(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def jobs2times(jobs):
    start_times = [0] * len(jobs)
    for j in jobs:
        start_times[j.id] = j.start_time
    return start_times

# --- DataClass Definitions ---
@dataclass
class SolutionInfo:
    makespan: int
    start_times: List[int]

@dataclass
class ResultInfo:
    problem_id: str
    best: SolutionInfo
    best_history: List[int]
    population_diversity: List[float]
    unique_solutions: List[int]
    scout_bees: List[int]

# --- Helper class for custom JSON encoding/decoding ---
class EnhancedJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that handles dataclasses.
    """
    def default(self, o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)

def _dataclass_from_dict(cls, data_dict):
    """
    Helper function to recursively convert dictionaries to dataclasses.
    """
    field_values = {}
    for field_name, field_type in cls.__annotations__.items():
        if field_name in data_dict:
            value = data_dict[field_name]
            # Check if the field type is a List of dataclasses
            if hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                if len(field_type.__args__) > 0 and is_dataclass(field_type.__args__[0]):
                    element_type = field_type.__args__[0]
                    field_values[field_name] = [_dataclass_from_dict(element_type, item) for item in value]
                else:
                    field_values[field_name] = value
            # Check if the field type is a dataclass itself
            elif is_dataclass(field_type):
                field_values[field_name] = _dataclass_from_dict(field_type, value)
            else:
                field_values[field_name] = value
        # else:
            # Handle missing fields if necessary, e.g., raise an error or use a default
            # print(f"Warning: Field '{field_name}' not found in dictionary for class {cls.__name__}")
    return cls(**field_values)


# --- Save and Load Functions ---
def save_results(results: List[ResultInfo], filename: str) -> None:
    """
    Saves a list of ResultInfo objects to a JSON file.

    Args:
        results: A list of ResultInfo objects to save.
        filename: The name of the file to save the results to.
                  (e.g., "optimization_results.json")
    """
    try:
        with open(filename, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=4, cls=EnhancedJSONEncoder)
        print(f"Results successfully saved to {filename}")
    except IOError as e:
        print(f"Error saving results to {filename}: {e}")
    except TypeError as e:
        print(f"Error serializing results: {e}")

def load_results(filename: str) -> List[ResultInfo]:
    """
    Loads a list of ResultInfo objects from a JSON file.

    Args:
        filename: The name of the file to load the results from.

    Returns:
        A list of ResultInfo objects, or an empty list if loading fails.
    """
    results: List[ResultInfo] = []
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            # Reconstruct the dataclasses from the dictionaries
            for item_dict in data:
                # Reconstruct the main SolutionInfo object for 'best'
                best_solution_reconstructed = None
                if 'best' in item_dict and item_dict['best'] is not None:
                     best_solution_reconstructed = SolutionInfo(**item_dict['best'])

                results.append(ResultInfo(
                    problem_id=item_dict.get('problem_id', ''), # Provide default if key missing
                    best=best_solution_reconstructed,
                    best_history=item_dict.get('best_history', []),
                    population_diversity=item_dict.get('population_diversity', []),
                    unique_solutions=item_dict.get('unique_solutions', []),
                    scout_bees=item_dict.get('scout_bees', [])
                ))
        print(f"Results successfully loaded from {filename}")
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filename}: {e}")
    except TypeError as e:
        print(f"Error reconstructing dataclasses (likely a mismatch in structure): {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading results: {e}")
    return results

def loadProblems(path):
    return [ProjectSchedulingModel.from_file(f"{path}{i}_{j}.sm") for i in range(1, 49) for j in range(1, 11)] 

def loadBestKnown(dataset):
    if dataset == 30:
        with open("j30opt.sm", "r") as f:
            lines = f.readlines()

        optimal = []
        for line in lines[22:502]:
            line = line.split()
            optimal.append(int(line[2]))
        
        return optimal
    
    with open(f"j{dataset}hrs.sm") as f:
        lines = f.readlines()
        
    optimal = []
    for line in lines[4:484]:
        line = line.split()
        optimal.append(int(line[2]))

    return optimal
