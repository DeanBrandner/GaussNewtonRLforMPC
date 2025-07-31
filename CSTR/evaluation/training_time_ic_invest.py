import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import pickle

if __name__ == "__main__":

    basepath = os.path.join("CSTR", "data", "ic_investigation")

    nIC = 100

    n_initial_guesses = 5

    # Load the agents trained with different algorithms

    # Gauss-Newton
    lr_gauss_newton = 1e-1
    beta_gauss_newton = 0.75
    eta_gauss_newton = 0.9
    omegainv_gauss_newton = 1e2

    gauss_newton_dict = {}
    for idx_initial_guess in tqdm(iterable=range(n_initial_guesses), desc="Loading training times for Gauss-Newton"):
        data_path = os.path.join(
            basepath,
            "gauss_newton",
            f"lr{lr_gauss_newton:.1e}_nIC{nIC}_beta_{beta_gauss_newton:.2f}_eta_{eta_gauss_newton:.3f}_omegainv_{omegainv_gauss_newton:.1e}",
            f"initial_guess_{idx_initial_guess}",
            "training_time.pkl"
        )
        initial_guess_dict = {}
        if os.path.exists(data_path):
            with open(data_path, "rb") as f:
                training_time_per_iteration = pickle.load(f)

            initial_guess_dict["time_per_iteration"] = training_time_per_iteration,
            initial_guess_dict["average_time_per_iteration"]= np.mean(training_time_per_iteration)
            initial_guess_dict["std_time_per_iteration"]= np.std(training_time_per_iteration)

        gauss_newton_dict[f"initial_guess_{idx_initial_guess + 1}"] = initial_guess_dict


    # Approximate Newton
    lr_approx_newton = 1e-1
    beta_approx_newton = 0.75
    eta_approx_newton = 0.9
    omegainv_approx_newton = 1e2

    approx_newton_dict = {}
    for idx_initial_guess in tqdm(iterable=range(n_initial_guesses), desc="Loading training times for Approximate Newton"):
        data_path = os.path.join(
            basepath,
            "approx_newton",
            f"lr{lr_approx_newton:.1e}_nIC{nIC}_beta_{beta_approx_newton:.2f}_eta_{eta_approx_newton:.3f}_omegainv_{omegainv_approx_newton:.1e}",
            f"initial_guess_{idx_initial_guess}",
            "training_time.pkl"
        )
        initial_guess_dict = {}
        if os.path.exists(data_path):
            with open(data_path, "rb") as f:
                training_time_per_iteration = pickle.load(f)

            initial_guess_dict["time_per_iteration"] = training_time_per_iteration,
            initial_guess_dict["average_time_per_iteration"]= np.mean(training_time_per_iteration)
            initial_guess_dict["std_time_per_iteration"]= np.std(training_time_per_iteration)

        approx_newton_dict[f"initial_guess_{idx_initial_guess + 1}"] = initial_guess_dict


    # Adam
    lr_adam = 1e-3
    adam_beta_1 = 0.9
    adam_beta_2 = 0.999

    adam_dict = {}
    for idx_initial_guess in tqdm(iterable=range(n_initial_guesses), desc="Loading training times for Gradient Ascent Adam"):
        data_path = os.path.join(
            basepath,
            "GA_Adam",
            f"lr{lr_adam:.1e}_nIC{nIC}_beta_1_{adam_beta_1:.2f}_beta_2_{adam_beta_2:.3f}",
            f"initial_guess_{idx_initial_guess}",
            "training_time.pkl"
        )
        initial_guess_dict = {}
        if os.path.exists(data_path):
            with open(data_path, "rb") as f:
                training_time_per_iteration = pickle.load(f)

            initial_guess_dict["time_per_iteration"] = training_time_per_iteration,
            initial_guess_dict["average_time_per_iteration"]= np.mean(training_time_per_iteration)
            initial_guess_dict["std_time_per_iteration"]= np.std(training_time_per_iteration)
        adam_dict[f"initial_guess_{idx_initial_guess + 1}"] = initial_guess_dict



    # Combine all methods into a single dictionary
    method_dict = {
        "Adam": adam_dict,
        "approx_newton": approx_newton_dict,
        "gauss_newton": gauss_newton_dict,
    }   


    # Print the computation times
    for method, initial_guesses in method_dict.items():
        print(f"Method: {method}")

        all_time_steps = []
        sum_time_of_this_initial_guess = []
        average_time_per_initial_guess = []
        std_time_per_initial_guess = []

        for initial_guess, results in initial_guesses.items():
            if "time_per_iteration" in results.keys():
                all_time_steps.extend(*results["time_per_iteration"])
                sum_of_this_initial_guess = np.sum(results["time_per_iteration"])
                average_of_this_initial_guess = np.mean(results["time_per_iteration"])
                std_of_this_initial_guess = np.std(results["time_per_iteration"])
                
                sum_time_of_this_initial_guess.append(sum_of_this_initial_guess)
                average_time_per_initial_guess.append(average_of_this_initial_guess)
                std_time_per_initial_guess.append(std_of_this_initial_guess)
                
                print(f"{initial_guess}: \t Average time per iteration {average_of_this_initial_guess:.4f} ± {std_of_this_initial_guess:.4f} seconds per iteration")
            else:
                print(f"{initial_guess}: \t No data available")

        method_dict[method]["all_time_steps"] = all_time_steps
        method_dict[method]["sum_time_of_this_initial_guess"] = sum_time_of_this_initial_guess
        method_dict[method]["average_time_per_initial_guess"] = average_time_per_initial_guess
        method_dict[method]["std_time_per_initial_guess"] = std_time_per_initial_guess
        print("\n")

    results_dict = {
        "average_time_per_iteration [s]": [],
        "std_time_per_iteration [s]": [],
        "average_time_per_iteration [min]": [],
        "std_time_per_iteration [min]": [],
        "average_time_per_initial_guess [s]": [],
        "std_time_per_initial_guess [s]": [],
        "average_time_per_initial_guess [min]": [],
        "std_time_per_initial_guess [min]": [],
    }
    index = []
    for key, value in method_dict.items():
        print(f"{key}:")

        average_time_per_iteration = np.mean(value["all_time_steps"])
        std_time_per_iteration = np.std(value["all_time_steps"])

        average_time_per_initial_guess = np.mean(value["sum_time_of_this_initial_guess"])
        std_time_per_initial_guess = np.std(value["sum_time_of_this_initial_guess"])

        print(f"Average time per iteration: {average_time_per_iteration:.4f} ± {std_time_per_iteration:.4f} seconds")
        print(f"Average time per iteration: {average_time_per_iteration/60:.4f} ± {std_time_per_iteration/60:.4f} min")
        print(f"Average time per initial guess: {average_time_per_initial_guess:.4f} ± {std_time_per_initial_guess:.4f} seconds")
        print(f"Average time per initial guess: {average_time_per_initial_guess/60:.4f} ± {std_time_per_initial_guess/60:.4f} min")

        index.append(key)
        results_dict["average_time_per_iteration [s]"].append(average_time_per_iteration)
        results_dict["std_time_per_iteration [s]"].append(std_time_per_iteration)
        results_dict["average_time_per_iteration [min]"].append(average_time_per_iteration / 60)
        results_dict["std_time_per_iteration [min]"].append(std_time_per_iteration / 60)

        results_dict["average_time_per_initial_guess [s]"].append(average_time_per_initial_guess)
        results_dict["std_time_per_initial_guess [s]"].append(std_time_per_initial_guess)
        results_dict["average_time_per_initial_guess [min]"].append(average_time_per_initial_guess / 60)
        results_dict["std_time_per_initial_guess [min]"].append(std_time_per_initial_guess / 60)

    computation_time_df = pd.DataFrame(data = results_dict, index = index)
    computation_time_df.to_csv(os.path.join(basepath, "training_time_summary.csv"))
    computation_time_df.to_excel(os.path.join(basepath, "training_time_summary.xlsx"))

    pass



