from typing import List
import random
from SchedulingProblem import Job
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy


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

import numpy as np

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

    
def normalize(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]
