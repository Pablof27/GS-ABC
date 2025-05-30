# Original imports
from Utils import loadProblems, loadBestKnown, save_results, ResultInfo, SolutionInfo, jobs2times
from ArtificialBeeColony import ArtificialBeeColony, Parameters
import os
import concurrent.futures

# Added import for progress bar
from tqdm import tqdm

PATHS = ["j30.sm/j30", "j60.sm/j60", "j90.sm/j90", "j120.sm/j120"]

# Helper function for a single optimization run (to be used with ProcessPoolExecutor)
# This function will be executed in a separate process.
def run_single_abc_optimization(problem_instance, params_obj, mode_str, init_str):
    """
    Runs a single ABC optimization for a given problem and parameters.
    This function is designed to be called by ProcessPoolExecutor.
    """
    # Imports needed by the worker process.
    # These are re-imported here to ensure availability in the new process,
    # especially for cross-platform compatibility (e.g., Windows) or different
    # multiprocessing start methods.
    from ArtificialBeeColony import ArtificialBeeColony # Assuming Parameters is defined here or imported by it
    from Utils import ResultInfo, SolutionInfo, jobs2times

    # Create and run the ABC algorithm
    abc = ArtificialBeeColony(psmodel=problem_instance)
    res = abc.optimize(params=params_obj, mode=mode_str, init=init_str)
    
    # Construct and return the result object
    return ResultInfo(
        problem_id=problem_instance.name,
        best=SolutionInfo(
            makespan=res.get_makespan(),
            start_times=jobs2times(res.jobs)
        ),
        best_history=[sol.get_makespan() for sol in abc.history],
        population_diversity=abc.population_divsersity,
        unique_solutions=abc.nunique_individuals,
        scout_bees=abc.nscout_bees
    )

def experiment_parallel(problems_list, filename, base_params, mode_str="basic", init_str="random", num_runs_per_problem=5, max_workers=None):
    """
    Runs experiments in parallel using ProcessPoolExecutor with a tqdm progress bar.
    Collects all results for the given problems_list and saves them once at the end.

    Args:
        problems_list (list): List of problem instances to run.
        filename (str): Filename to save the results.
        base_params (Parameters): Parameters object for the ABC algorithm.
        mode_str (str): Mode for the ABC algorithm ('basic', 'variant', etc.).
        init_str (str): Initialization strategy for the ABC algorithm ('random', 'mcmc', etc.).
        num_runs_per_problem (int): Number of times to run the algorithm for each problem instance.
        max_workers (int, optional): Maximum number of worker processes. Defaults to os.cpu_count().
    """
    all_results = []
    
    # Create a list of all individual tasks to be run.
    # Each task consists of the arguments needed by run_single_abc_optimization.
    tasks_to_submit = []
    for p_instance in problems_list:
        for _ in range(num_runs_per_problem):
            tasks_to_submit.append((p_instance, base_params, mode_str, init_str))
            
    if not tasks_to_submit:
        print(f"No tasks to run for {filename}. Skipping.")
        return []

    print(f"Starting parallel execution for {filename} with {len(tasks_to_submit)} total runs.")
    
    # Use ProcessPoolExecutor for parallel execution.
    # If max_workers is None, it defaults to the number of processors on the machine.
    if max_workers is None:
        max_workers = os.cpu_count()
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor.
        # executor.submit(fn, *args, **kwargs)
        futures = [executor.submit(run_single_abc_optimization, p_inst, params_obj, m_str, i_str)
                   for p_inst, params_obj, m_str, i_str in tasks_to_submit]

        # Use tqdm for a progress bar as tasks complete.
        # concurrent.futures.as_completed yields futures as they finish.
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Optimizing {filename}"):
            try:
                result = future.result()  # Get the ResultInfo object from the completed future
                all_results.append(result)
            except Exception as exc:
                # Handle exceptions from individual tasks if necessary
                # For now, just print the exception. You might want to log more details
                # or store information about which specific task failed.
                print(f"\nA task for {filename} generated an exception: {exc}")
                # Optionally, append a placeholder or error marker to all_results
                # or re-raise if critical.

    # Save all collected results once after all parallel tasks are done.
    if all_results:
        # Ensure Utils.save_results is available (it's imported at the top level)
        save_results(results=all_results, filename=filename)
        print(f"Results for {filename} saved successfully. Total results: {len(all_results)}.")
    else:
        print(f"No results were collected for {filename}.")
    
    return all_results # Optionally return the collected results

# Main execution block
if __name__ == "__main__":
    # Ensure that custom modules (Utils, ArtificialBeeColony) are accessible.
    # This is typically true if they are in the same directory or in PYTHONPATH.

    print("Loading problems and best known solutions...")
    # Load problem sets
    problems_data = {path.split(".")[0]: loadProblems(path) for path in PATHS}
    # Load best known solutions
    best_known_data = {f"j{i}": loadBestKnown(i) for i in [30, 60, 90, 120]}
    
    # Assign best_known solutions to problem instances
    for key in problems_data.keys():
        if key in best_known_data:
            current_problems = problems_data[key]
            current_best_known = best_known_data[key]
            for i, best_val in enumerate(current_best_known):
                if i < len(current_problems):
                    current_problems[i].best_known = best_val
                else:
                    # This case should ideally not happen if data is consistent
                    print(f"Warning: More best_known values than problems for {key}. Index {i} is out of bounds for problems list (length {len(current_problems)}).")
        else:
            print(f"Warning: No best_known data found for problem key {key}.")


    # Define algorithm parameters
    params_dabc_obj = Parameters(N=200, limit=100, max_trials=500, mr=0.0)
    params_gsabc_obj = Parameters(N=200, limit=100, max_trials=500, mr=0.1, l=50) # Assuming 'l' is a valid parameter
    
    # Determine max_workers for the ProcessPoolExecutor
    # Uses all available CPU cores by default if None is passed to experiment_parallel or ProcessPoolExecutor
    num_cores = os.cpu_count()
    print(f"Configured to use up to {num_cores} cores for parallel execution.")

    # --- Run experiments for j30 problem set ---
    print("\nStarting experiments for j30 problems...")
    experiment_parallel(
        problems_list=problems_data.get("j30", []), # Use .get for safety if a key might be missing
        filename="dabc_j30_parallel.json",    # Changed filename to denote parallel execution
        base_params=params_dabc_obj,
        mode_str="basic", 
        init_str="random",
        max_workers=num_cores  # Explicitly pass num_cores
    )
    
    # experiment_parallel(
    #     problems_list=problems_data.get("j30", []),
    #     filename="gsabc_j30_parallel.json",   # Changed filename
    #     base_params=params_gsabc_obj, 
    #     mode_str="variant", 
    #     init_str="mcmc",
    #     max_workers=num_cores
    # )
    
    # --- Optionally, run experiments for other problem sets (j60, j90, j120) ---
    # You can uncomment and adapt these sections to run more experiments.

    # print("\nStarting experiments for j60 problems...")
    # experiment_parallel(
    #     problems_list=problems_data.get("j60", []),
    #     filename="dabc_j60_parallel.json",
    #     base_params=params_dabc_obj,
    #     mode_str="basic",
    #     init_str="random",
    #     max_workers=num_cores
    # )
    # experiment_parallel(
    #     problems_list=problems_data.get("j60", []),
    #     filename="gsabc_j60_parallel.json",
    #     base_params=params_gsabc_obj,
    #     mode_str="variant",
    #     init_str="mcmc",
    #     max_workers=num_cores
    # )

    # print("\nStarting experiments for j90 problems...")
    # # ... (similar calls for j90)

    # print("\nStarting experiments for j120 problems...")
    # # ... (similar calls for j120)

    print("\nAll scheduled experiments have been initiated.")
    print("Note: Actual completion depends on the execution time of the tasks.")
