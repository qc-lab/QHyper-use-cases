from QHyper.problems import KnapsackProblem
from QHyper.solvers import solver_from_config
from QHyper.util import (
    weighted_avg_evaluation, sort_solver_results, add_evaluation_to_results)

import numpy as np

if __name__ == '__main__':
    problem_config = {  # This is going to be used later
        'type': 'knapsack',
        'max_weight': 5,
        'items': [(1, 2), (2, 4), (1, 2), (1, 1), (4, 5), (1, 2), (3, 1)]
    }

    params_config = {
        'angles': [[0.5]*5, [1]*5],
        'hyper_args': [1, 2.5, 2.5],
    }
    hyper_optimizer_bounds = 3*[(1, 10)]
    penalty = 5


    solver_config = {
        "solver": {
            "type": "vqa",
            "optimizer": {
                # "type": "scipy",
                # "maxfun": 200,

                "type": "random",
                "number_of_samples": 1,
                "bounds": 10*[(0, 2*np.pi)],
                # "type": "cem",
                "processes": 1,
                # "samples_per_epoch": 500,
                # "epochs": 5,
                # "bounds": 10*[(0, 2*np.pi)],
                "verbose": True,
            },
            "pqc": {
                "type": "qaoa",
                "layers": 5,
                # "limit_results": 20,b
                # "penalty": penalty,
            },
            "params_inits": params_config,
            # "hyper_optimizer": {
            #     "type": "cem",
            #     "processes": 5,
            #     "samples_per_epoch": 5000,
            #     "epochs": 5,
            #     "bounds": hyper_optimizer_bounds,
            # }
        },
        "problem": problem_config
    }


    from QHyper.experiment_utils import Experiment


    experiment = Experiment()
    results = experiment.run_solver_multiple_times(solver_config, 1000, 10)

    print(results)

    mean = np.mean(results)
    std = np.std(results)

    print(f"Mean: {mean}, std: {std}")

