from Utils import loadProblems, loadBestKnown, save_results, ResultInfo, SolutionInfo, jobs2times
from ArtificialBeeColony import ArtificialBeeColony, Parameters
import os
import concurrent.futures

PATHS = ["j30.sm/j30", "j60.sm/j60", "j90.sm/j90", "j120.sm/j120"]

def experiment(problems, filename, params, mode="basic", init="random"):
    results = []
    for i, p in enumerate(problems):
        for _ in range(5):
            abc = ArtificialBeeColony(psmodel=p)
            res = abc.optimize(params=params, mode=mode, init=init)
            results.append(
                ResultInfo(
                    problem_id=p.name,
                    best=SolutionInfo(
                        makespan=res.get_makespan(),
                        start_times=jobs2times(res.jobs)
                    ),
                    best_history=[sol.get_makespan() for sol in abc.history],
                    population_diversity=abc.population_divsersity,
                    unique_solutions=abc.nunique_individuals,
                    scout_bees=abc.nscout_bees
                )
            )
        save_results(results=results, filename=filename)
    
if __name__ == "__main__":
    problems = {path.split(".")[0]: loadProblems(path) for path in PATHS}
    best_known = {f"j{i}": loadBestKnown(i) for i in [30, 60, 90, 120]}
    
    for key in problems.keys():
        for i, best in enumerate(best_known[key]):
            problems[key][i].best_known = best

    params_dabc = Parameters(N=200, limit=100, max_trials=500, mr=0.0)
    params_gsabc = Parameters(N=200, limit=100, max_trials=500, mr=0.1, l=50)
    
    experiment(problems=problems["j30"], filename="dabc_j30.json", params=params_dabc)
    experiment(problems=problems["j30"], filename="gsabc_j30.json", params=params_gsabc, mode="variant", init="mcmc")

    