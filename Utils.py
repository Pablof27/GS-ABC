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
        sorted_jobs.append(job)
        
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

def plot_multiple_schedules(schedules, titles=None, mapping=None, save_name=""):
    num_schedules = len(schedules)
    _, num_times = schedules[0].shape  # All schedules have the same number of columns (time steps)

    FONT_SIZE = 10
    
    plt.rcParams.update({
        'font.family': 'Courier New',
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'figure.titlesize': FONT_SIZE
    })

    # Create a figure with subplots
    fig, axes = plt.subplots(num_schedules, 1, figsize=(10, 2 * num_schedules), squeeze=False)

    # Create a color map for different jobs (shared across all schedules)
    unique_jobs = np.unique(np.concatenate([np.unique(schedule) for schedule in schedules]))
    color_map = plt.get_cmap('viridis', len(unique_jobs))
    
    # supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
    shuffle_map = mapping if mapping is not None else np.random.permutation(len(unique_jobs))

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
        ax.set_ylabel(f'R{k+1}')
        if titles is not None and k < len(titles):
            ax.set_title(titles[k])
        ax.set_xticks(np.arange(0, num_times + 1, 1))
        ax.set_yticks(np.arange(0, schedule.shape[0] + 1, 1))
        ax.set_yticklabels([])
        ax.grid(True, which='major', linestyle='-', linewidth=0.25, alpha=0.5)

    plt.tight_layout()
    if save_name != "":
        plt.savefig(f"{save_name}.svg")
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

    # with open(f"j{dataset}lb.sm") as f:
    #     lines = f.readlines()
    
    # lower_bounds = []
    # for line in lines[11:]:
    #     pass

    return optimal

def arpd(results, problems: list[ProjectSchedulingModel]):
    desviations = {problem.name: [] for problem in problems}
    desviation_sum = 0
    for result in results:
        problem = next(filter(lambda x: x.name == result.problem_id, problems))
        desviations[problem.name].append(result.best.makespan - problem.best_known)
        desviation_sum += desviations[problem.name][-1]

    return desviation_sum/len(desviations), desviations

def plot_multiple_schedules_with_legend(schedules, titles=None, colormap_name='viridis', job_id_mapping=None):
    """
    Plots multiple schedules with a shared legend for job IDs.

    Args:
        schedules (list of np.ndarray): A list of 2D numpy arrays. Each array represents a schedule
                                         where rows are resources and columns are time steps.
                                         Cell values are job IDs, -1 typically represents idle.
        titles (list of str, optional): A list of titles for each subplot. Defaults to None.
        colormap_name (str, optional): Name of the matplotlib colormap to use. Defaults to 'viridis'.
        job_id_mapping (list or np.ndarray, optional): A specific permutation of color indices
                                                      [0, ..., num_unique_jobs-1] to map jobs to colors.
                                                      If None, a random permutation is used. Defaults to None.
    """
    if not schedules:
        print("No schedules provided to plot.")
        return

    num_schedules = len(schedules)
    # Assuming all schedules have the same number of time steps, taken from the first schedule
    # If schedules can have different num_times, this needs to be handled (e.g., max_num_times)
    if schedules[0].ndim != 2 or schedules[0].shape[1] == 0:
        print("First schedule is invalid (not 2D or no time steps). Cannot determine number of time steps.")
        return
    _, num_times = schedules[0].shape

    # Create a figure with subplots
    # Adjusted figsize slightly, and tight_layout will manage spacing with legend
    fig, axes = plt.subplots(num_schedules, 1, figsize=(15, 2 * num_schedules + 1), squeeze=False)

    # Consolidate all job IDs from all schedules, excluding -1 (idle)
    all_job_ids_list = []
    for schedule in schedules:
        if schedule.ndim == 2: # Basic check for valid schedule format
            all_job_ids_list.extend(np.unique(schedule[schedule != -1]))
    
    if not all_job_ids_list:
        print("No actual jobs found in schedules (all idle or schedules are empty/invalid).")
        # Plot empty axes if needed, basic setup
        for k in range(num_schedules):
            ax = axes[k, 0]
            ax.set_xlabel('Time')
            ax.set_ylabel(f'Resource {k+1}')
            if titles is not None and k < len(titles):
                ax.set_title(titles[k])
            
            current_schedule_shape = schedules[k].shape if schedules[k].ndim == 2 else (0, num_times)
            num_resources_this_schedule = current_schedule_shape[0]

            ax.set_xlim(0, num_times)
            ax.set_ylim(0, max(1, num_resources_this_schedule)) # Ensure ylim is not (0,0)
            ax.set_xticks(np.arange(0, num_times + 1, 1))
            ax.set_yticks(np.arange(0, max(1, num_resources_this_schedule) + 1, 1))
            ax.set_yticklabels([])
            ax.grid(True, which='major', linestyle='-', linewidth=0.25, alpha=0.5)
            if num_resources_this_schedule > 0 : ax.invert_yaxis()
        if num_schedules > 0: fig.tight_layout()
        plt.show()
        return

    actual_unique_job_ids = np.unique(all_job_ids_list)
    num_distinct_jobs = len(actual_unique_job_ids)

    if num_distinct_jobs == 0: # Should be caught by previous check, but as a safeguard
        print("No unique jobs to plot after filtering.")
        # Similar empty plot logic as above if needed
        if num_schedules > 0: fig.tight_layout()
        plt.show()
        return

    # Create a mapping from actual job ID to a 0-based sequential index
    job_id_to_sequential_idx = {job_id: i for i, job_id in enumerate(actual_unique_job_ids)}

    # Create a color map for different jobs
    try:
        color_palette = plt.get_cmap(colormap_name, num_distinct_jobs)
    except ValueError:
        print(f"Colormap '{colormap_name}' not recognized or num_distinct_jobs is zero. Using 'viridis'.")
        color_palette = plt.get_cmap('viridis', num_distinct_jobs if num_distinct_jobs > 0 else 1)


    # This is a permutation of indices [0, 1, ..., num_distinct_jobs-1]
    permuted_color_indices = job_id_mapping
    if permuted_color_indices is None:
        permuted_color_indices = np.random.permutation(num_distinct_jobs)
    elif len(permuted_color_indices) != num_distinct_jobs:
        print(f"Warning: Provided job_id_mapping length ({len(permuted_color_indices)}) "
              f"does not match number of unique jobs ({num_distinct_jobs}). Using random permutation.")
        permuted_color_indices = np.random.permutation(num_distinct_jobs)

    # Plot each schedule in a separate subplot
    for k, schedule_data in enumerate(schedules):
        ax = axes[k, 0]
        
        if schedule_data.ndim != 2:
            print(f"Warning: Schedule {k} is not 2D. Skipping.")
            ax.text(0.5, 0.5, "Invalid schedule data", ha='center', va='center')
            continue # Skip to the next schedule

        num_resources_this_schedule = schedule_data.shape[0]

        for i in range(num_resources_this_schedule):  # Iterate over resources for this specific schedule
            for j in range(num_times): # Iterate over time steps
                current_job_id = schedule_data[i, j]
                if current_job_id != -1:  # Assuming -1 represents idle time
                    if current_job_id in job_id_to_sequential_idx:
                        base_idx = job_id_to_sequential_idx[current_job_id]
                        color_map_idx = permuted_color_indices[base_idx]
                        color = color_palette(color_map_idx)
                        # Add a patch for the job
                        ax.add_patch(patches.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='gray', linewidth=0.25))
                    else:
                        # This case means a job ID is in the schedule but not in actual_unique_job_ids
                        # This might happen if -1 was the only thing filtered initially but other negatives exist
                        # Or if actual_unique_job_ids was somehow incomplete.
                        # For robustness, plot with a default "unknown" color.
                        ax.add_patch(patches.Rectangle((j, i), 1, 1, facecolor='lightgrey', edgecolor='black', linewidth=0.25, hatch='//'))


        # Set labels and ticks for each subplot
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Resource {k+1}')
        if titles is not None and k < len(titles):
            ax.set_title(titles[k])
        
        ax.set_xlim(0, num_times)
        ax.set_ylim(0, num_resources_this_schedule if num_resources_this_schedule > 0 else 1)
        ax.set_xticks(np.arange(0, num_times + 1, 1))
        # Set y-ticks to be at the center of each resource row for clarity if labels were used
        # Or at the boundaries as it was. For now, keeping boundaries.
        ax.set_yticks(np.arange(0, num_resources_this_schedule + 1, 1))
        ax.set_yticklabels([]) # Keep Y-axis resource numbers abstract (no specific labels)
        ax.grid(True, which='both', linestyle='-', linewidth=0.2, alpha=0.7) # 'both' for minor ticks if they existed
        
        if num_resources_this_schedule > 0:
            ax.invert_yaxis() # Invert Y-axis so resource 0 is at the top

    # Create legend handles
    legend_handles = []
    # Sort unique job IDs for consistent legend order (if they are sortable)
    try:
        # Ensure job IDs are treated as fundamental types for sorting if they are numpy types
        sorted_job_ids_for_legend = sorted([job.item() if hasattr(job, 'item') else job for job in actual_unique_job_ids])
    except TypeError: 
        # Fallback if items are not directly sortable (e.g., mixed types not handled by default sort)
        sorted_job_ids_for_legend = list(actual_unique_job_ids)

    for job_id in sorted_job_ids_for_legend:
        if job_id not in job_id_to_sequential_idx: continue # Should not happen if sorted_job_ids_for_legend from actual_unique_job_ids

        base_idx = job_id_to_sequential_idx[job_id]
        color_map_idx = permuted_color_indices[base_idx]
        color = color_palette(color_map_idx)
        
        # Ensure label is a string
        label_text = f'Job {job_id.decode()}' if isinstance(job_id, bytes) else f'Job {str(job_id)}'
            
        legend_handles.append(patches.Patch(facecolor=color, edgecolor='gray', label=label_text))

    # Add legend to the figure
    if legend_handles:
        # Position legend to the right of the subplots
        fig.legend(handles=legend_handles, title="Job Legend", loc='center left', bbox_to_anchor=(1.0, 0.5))
        # Adjust layout to make space for the legend. rect is [left, bottom, right, top]
        fig.tight_layout(rect=[0, 0, 0.85, 1]) 
