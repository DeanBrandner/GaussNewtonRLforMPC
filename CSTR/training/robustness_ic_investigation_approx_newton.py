import os
import pickle
import casadi as cd
import pandas as pd
import numpy as np
import tensorflow as tf

from mpc_collection import get_rl_mpc
from mp_utils import run_episode_loop

from rl_mpc_agents import RL_MPC_AN_agent as Agent

from environments import CSTR as Environment
from multiprocessing import Process, Queue, Value

from tqdm import tqdm
import time


def train(n_processes:int, agent_path:str, rl_settings: dict = {}, n_episodes_per_replay: int = 50, initial_guess_idx: int = 0):

    mpc = get_rl_mpc()

    agent = Agent(mpc, rl_settings, init_differentiator = True)
    
    if initial_guess_idx > 0:
        param_lb = np.array([-0.05, -0.75]).reshape(-1, 1)
        param_ub = np.array([+0.07, +0.10]).reshape(-1, 1)
        initial_guess_rng = np.random.default_rng(initial_guess_idx)
        parameters = param_lb + (param_ub - param_lb) * initial_guess_rng.uniform(0, 1, size = mpc.p_fun(0).master.shape)
        parameters = cd.DM(parameters)
    else:
        parameters = mpc.p_fun(0).master

    p_template = agent.mpc.get_p_template(1)
    p_template.master = parameters
    mpc.set_p_fun(lambda t_now: p_template)

    task_queue = Queue()
    environment_queue = Queue()
    progress_counter = Value("i", 0)

    time_per_replay = []
    
    # Start the workers
    workers = []
    for i in range(n_processes):
        worker = Process(target = run_episode_loop, args = (get_rl_mpc, Agent, rl_settings, Environment, agent_path, task_queue, environment_queue, progress_counter))
        workers.append(worker)
        worker.start()

    agent.save(os.path.join(agent_path, f"agent_update"))
    agent.save_rl_parameters(os.path.join(agent_path, f"agent_update_0"))
    agent.save_q_func(os.path.join(agent_path, f"agent_update_0"))

    # Do the training
    n_replays = 101

    unprocessed_results_list= []
    processed_results_list = []

    pbar_replays = tqdm(desc = "Replays".rjust(20), total = n_replays)
    pbar_episodes = tqdm(desc = "Episodes".rjust(20), total = n_episodes_per_replay)
    for replay_idx in range(n_replays):

        start_time = time.time()
        pbar_replays.update(1)
        pbar_replays.refresh()

        with progress_counter.get_lock():
            progress_counter.value = 0
        pbar_episodes.n = 0
        pbar_episodes.reset()

        for seed in range(n_episodes_per_replay):
            task_queue.put({
                "episode": replay_idx,
                "seed": seed,
                "parameters": agent.mpc.p_fun(0).master,
                "max_steps_of_violation": 1,  # This can be adjusted based on the environment
                "training": True,
            })

        gathered_results = []
        for seed in range(n_episodes_per_replay):
            env_results = environment_queue.get()

            gathered_results.append(env_results)
            # monitor.append(env_results)

            with progress_counter.get_lock():
                completed = progress_counter.value
            pbar_episodes.n = completed
            pbar_episodes.refresh()

        agent.synchronize_memories(path = os.path.join(agent_path, "memories"))

        cum_reward_list = []
        stage_cost_list = []
        penalty_list = []
        termination_list = []
        truncation_list = []
        episode_time_list = []
        for result in gathered_results:
            data = result["data"]
            cum_reward_list.append(data.r.sum())
            stage_cost_list.append(data.stage_cost.sum())
            penalty_list.append(data.penalty.sum())
            termination_list.append(int(result["termination"]))
            truncation_list.append(int(result["truncation"]))
            episode_time_list.append(data.time[-1])

        unprocessed_results = pd.DataFrame(
            data = {
                "cum_reward": cum_reward_list,
                "stage_cost": stage_cost_list,
                "penalty": penalty_list,
                "termination": termination_list,
                "truncation": truncation_list,
                "episode_time": episode_time_list
            }
        )
        processed_results = unprocessed_results.describe()

        unprocessed_results_list.append(unprocessed_results)
        processed_results_list.append(processed_results)

        with open(os.path.join(agent_path, f"unprocessed_results_list.pkl"), "wb") as f:
            pickle.dump(unprocessed_results_list, f)
        with open(os.path.join(agent_path, f"processed_results_list.pkl"), "wb") as f:
            pickle.dump(processed_results_list, f)
        

        policy_gradient, policy_hessian = agent.replay()

        print(agent.mpc.p_fun(0)["_p"])

        agent.save_rl_parameters(os.path.join(agent_path, f"agent_update_{replay_idx + 1}"))
        agent.save_q_func(os.path.join(agent_path, f"agent_update_{replay_idx + 1}"))
        with open(os.path.join(agent_path, f"agent_update_{replay_idx}", "policy_gradients.pkl"), "wb") as f:
            pickle.dump(policy_gradient, f)
        with open(os.path.join(agent_path, f"agent_update_{replay_idx}", "policy_hessian.pkl"), "wb") as f:
            pickle.dump(policy_hessian, f)

        end_time = time.time()

        replay_time = end_time - start_time
        time_per_replay.append(replay_time)

        with open(os.path.join(agent_path, "training_time.pkl"), "wb") as f:
            pickle.dump(time_per_replay, f)

    pbar_replays.close()
    pbar_episodes.close()

    
    for worker in workers:
        task_queue.put("STOP")
    for worker in workers: 
        worker.join()

    return

if __name__ == "__main__":

    n_mpc_processes = min(int(os.cpu_count() // 2) - 1, 50)
    n_episodes_per_replay = 100

    n_initial_guesses_for_rl_params = 5

    actor_learning_rate = 1e-1
    use_momentum = True
    use_adam = False
    beta = 0.75
    eta = 0.9
    omegainv = 1e2


    rl_settings = {
            "gamma": 1.0,
            "exploration_noise": 1e-2 * np.abs(np.array([40 - 5, -8500 - 0])).reshape(-1, 1),
            "exploration_distribution": "normal",
            "exploration_seed": 1,
            "actor_learning_rate": actor_learning_rate,

            "q_func_architecture": [64, 32],
            "q_func_activation": "gelu",
            "q_func_optimizer": "adam",
            "q_func_learning_rate": 1e-3,
            "q_func_loss": "mse",
            "q_func_batch_size": 128,
            "q_func_epochs": 500,
            "q_func_always_reset": False,

            "clip_q_gradients": True,
            "clip_jac_policy": True,
            "use_momentum": use_momentum,
            "use_adam": use_adam,
            "momentum_beta": beta,
            "momentum_eta": eta,
            "omegainv": omegainv
        }


    for idx in range(n_initial_guesses_for_rl_params):
        agent_path = os.path.join("CSTR", "data", "ic_investigation", f"approx_newton", f"lr{actor_learning_rate:.1e}_nIC{n_episodes_per_replay}_beta_{beta:.2f}_eta_{eta:.3f}_omegainv_{omegainv:.1e}", f"initial_guess_{idx}")

        tf.keras.backend.clear_session()
                
        start_time = time.time()
        train(n_mpc_processes, agent_path, rl_settings, n_episodes_per_replay, idx)
        end_time = time.time()

        print(f"Training time: {end_time - start_time:.2f} seconds")
        print(f"Training time: {(end_time - start_time) / 60:.2f} minutes")
        print(f"Training time: {(end_time - start_time) / 3600:.2f} hours")
        print(f"Training time: {(end_time - start_time) / 86400:.2f} days")

        with open(os.path.join(agent_path, "training_time.txt"), "w") as f:
            f.write(f"Training time: {end_time - start_time:.2f} seconds\n")
            f.write(f"Training time: {(end_time - start_time) / 60:.2f} minutes\n")
            f.write(f"Training time: {(end_time - start_time) / 3600:.2f} hours\n")
            f.write(f"Training time: {(end_time - start_time) / 86400:.2f} days\n")