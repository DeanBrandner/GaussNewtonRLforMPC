# Import necessary modules and functions
from mpc_collection import get_rl_mpc
from environments import CSTR as Environment
from rl_mpc_agents import RL_MPC_agent as Agent
from mp_utils import run_episode_loop

from multiprocessing import Process, Queue, Value
from tqdm import tqdm
import os
import numpy as np
import casadi as cd
import pandas as pd

import pickle

from matplotlib import pyplot as plt
plt.switch_backend("agg")  # Use a non-interactive backend for matplotlib

def compute_parametric_performance(parametric_results_path:str, param_range_alpha:np.ndarray, param_range_beta:np.ndarray, n_initial_conditions:int, n_mpc_processes:int = 1):
    """
    Runs parametric performance evaluation for RL-MPC on CSTR environment over a grid of alpha/beta parameters.
    Stores raw and processed results, and generates performance plots.

    Args:
        parametric_results_path (str): Path to save results.
        param_range_alpha (np.ndarray): Range of alpha parameter values.
        param_range_beta (np.ndarray): Range of beta parameter values.
        n_initial_conditions (int): Number of initial conditions per parameter set.
        n_mpc_processes (int, optional): Number of parallel processes. Defaults to 1.

    Returns:
        None
    """
    # Queues and shared counter for multiprocessing
    task_queue = Queue()
    environment_queue = Queue()
    progress_counter = Value("i", 0)

    # Limit number of processes to available CPUs and initial conditions
    n_mpc_processes = min(n_mpc_processes, n_initial_conditions)
    n_mpc_processes = min(n_mpc_processes, os.cpu_count() // 2)
    workers = []
    # Start worker processes
    for i in range(n_mpc_processes):
        worker = Process(target = run_episode_loop, args = (get_rl_mpc, Agent, {}, Environment, parametric_results_path, task_queue, environment_queue, progress_counter))
        workers.append(worker)
        worker.start()

    # Create meshgrid for parameter sweep
    param_range_alpha_mesh,  param_range_beta_mesh = np.meshgrid(param_range_alpha, param_range_beta)

    # Progress bar for episodes
    pbar_episodes = tqdm(desc = "Episodes".rjust(20), total = n_initial_conditions, position = 2)

    # Reset progress counter and progress bar
    with progress_counter.get_lock():
        progress_counter.value = 0
    pbar_episodes.n = 0
    pbar_episodes.reset()

    # Allocate array for storing episode performance data
    performance_data_per_episode = np.empty(shape = param_range_alpha_mesh.shape, dtype=object)

    # Loop over all parameter combinations
    for idx in tqdm(iterable = range(param_range_alpha_mesh.shape[0]), desc = "beta", position = 0):
        for jdx in tqdm(iterable = range(param_range_alpha_mesh.shape[1]), desc = "alpha", position = 1):

            alpha_value = param_range_alpha_mesh[idx, jdx]
            beta_value = param_range_beta_mesh[idx, jdx]

            param_value = cd.vertcat(alpha_value, beta_value)

            # Enqueue tasks for each initial condition
            for seed in range(n_initial_conditions):
                task_queue.put({
                    "seed": seed,
                    "parameters": param_value,
                })

            # Reset progress counter and bar for this parameter set
            with progress_counter.get_lock():
                progress_counter.value = 0
            performance_data = []
            pbar_episodes.n = 0
            pbar_episodes.refresh()
            # Collect results from environment queue
            for seed in range(n_initial_conditions):
                env_results = environment_queue.get()
                performance_data.append(env_results)
                with progress_counter.get_lock():
                    completed = progress_counter.value
                pbar_episodes.n = completed
                pbar_episodes.refresh()

            performance_data_per_episode[idx, jdx] = performance_data

    pbar_episodes.close()

    # Stop and join worker processes
    for worker in workers:
        task_queue.put("STOP")
    for worker in workers: 
        worker.join()

    # Save raw episode data
    with open(os.path.join(parametric_results_path, "performance_data_per_episode.pkl"), "wb") as f:
        pickle.dump(performance_data_per_episode, f)
    
    # Analyze and summarize performance data
    unprocessed_results_mesh = np.empty(shape = performance_data_per_episode.shape, dtype=object)
    for idx in range(performance_data_per_episode.shape[0]):
        for jdx in range(performance_data_per_episode.shape[1]):
            performance_data = performance_data_per_episode[idx, jdx]
    
            # Lists to collect statistics for each episode
            episodes = []
            seeds = []
            returns = []
            stage_costs = []
            penalties = []
            episode_lengths = []
            total_number_of_cv = []
            total_number_of_cv_episodes =0
            total_number_of_points = 0
            termination = []
            truncation = []

            # Extract statistics from each initial condition
            for ic_data in tqdm(iterable = performance_data, desc= "Plotting"):
                episodes.append(ic_data["episode"])
                seeds.append(ic_data["seed"])
                data = ic_data["data"]

                returns.append(data.r.sum())
                stage_costs.append(data.stage_cost.sum())
                penalties.append(data.penalty.sum())
                episode_lengths.append(data.r.shape[0])
                local_number_of_cv = (data.penalty > 1e-6).sum()
                total_number_of_points += data.r.shape[0]
                total_number_of_cv.append(local_number_of_cv)
                total_number_of_cv_episodes += 1

                termination.append(int(ic_data["termination"]))
                truncation.append(int(ic_data["truncation"]))

            # Store statistics in DataFrame
            unprocessed_results = pd.DataFrame(
                data = {
                    "episode": episodes,
                    "seed": seeds,
                    "return": returns,
                    "stage_cost": stage_costs,
                    "penalty": penalties,
                    "episode_length": episode_lengths,
                    "total_number_of_cv": total_number_of_cv,
                    "termination": termination,
                    "truncation": truncation,
                }
            )
            unprocessed_results_mesh[idx, jdx] = unprocessed_results

    # Save unprocessed results
    with open(os.path.join(parametric_results_path, "unprocessed_results_mesh.pkl"), "wb") as f:
        pickle.dump(unprocessed_results_mesh, f)
    
    # Compute summary statistics for each parameter combination
    processed_results_mesh = np.empty(shape = unprocessed_results_mesh.shape, dtype=object)
    for idx in range(unprocessed_results_mesh.shape[0]):
        for jdx in range(unprocessed_results_mesh.shape[1]):
            unprocessed_results = unprocessed_results_mesh[idx, jdx]
            processed_results = unprocessed_results.describe()
            processed_results_mesh[idx, jdx] = processed_results
        
    # Save processed results and parameter grids
    with open(os.path.join(parametric_results_path, "processed_results_mesh.pkl"), "wb") as f:
        pickle.dump(processed_results_mesh, f)
    
    with open(os.path.join(parametric_results_path, "param_range_alpha_mesh.pkl"), "wb") as f:
        pickle.dump(param_range_alpha_mesh, f)
    with open(os.path.join(parametric_results_path, "param_range_beta_mesh.pkl"), "wb") as f:
        pickle.dump(param_range_beta_mesh, f)
    
    return

if __name__ == "__main__":
    # Define parameter ranges for medium parameterization
    param_medium_alpha_beta_range_alpha = np.linspace(-0.05, 0.10, 2) # Placeholder for medium parameterization
    param_medium_alpha_beta_range_beta = np.linspace(-0.75, 0.1, 2)  # Placeholder for medium parameterization

    # Set number of processes and initial conditions
    n_mpc_processes = min(os.cpu_count() // 2 - 1, 50)
    n_initial_conditions = 100
    parametric_results_path = os.path.join("CSTR", "data", "parametric_results", f"n_initial_conditions_{n_initial_conditions}")

    # Run full parametric performance computation
    compute_parametric_performance(
        parametric_results_path,
        param_medium_alpha_beta_range_alpha,
        param_medium_alpha_beta_range_beta,
        n_initial_conditions,
        n_mpc_processes,
    )