import os, pickle, time, gc
import pandas as pd
import numpy as np

from multiprocessing import Process, Queue, Value
from tqdm import tqdm

from mp_utils import run_episode_loop
from rl_mpc_agents import RL_MPC_AN_agent as Agent
from environments import CSTR as Environment

global_seed = 42


def train(
        n_processes:int,
        penalty_weights,
        agent_path:str,
        rl_settings: dict = {},
        n_episodes_per_replay: int = 50,
        parameterization: str = "low",
        ):

    if parameterization == "low":
        from mpc_collection import get_rl_mpc
    elif parameterization == "medium":
        from mpc_collection import get_rl_mpc_medium_parameterized as get_rl_mpc
    elif parameterization == "high":
        from mpc_collection import get_rl_mpc_high_parameterized as get_rl_mpc
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    mpc = get_rl_mpc(penalty_weights)

    agent = Agent(mpc, rl_settings, init_differentiator = True)
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
        worker = Process(target = run_episode_loop, args = (get_rl_mpc, penalty_weights, Agent, rl_settings, Environment, agent_path, task_queue, environment_queue, progress_counter))
        workers.append(worker)
        worker.start()

    agent.save(os.path.join(agent_path, f"agent_update"))
    agent.save_rl_parameters(os.path.join(agent_path, f"agent_update_0"))

    # Do the training
    n_replays = 51

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
                "measurement_noise": False,
                "additive_process_noise": True,
                "parametric_uncertainty": True,
                "diff_twice": True
            })

        gathered_results = []
        for seed in range(n_episodes_per_replay):
            env_results = environment_queue.get()

            gathered_results.append(env_results)

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
            cum_reward = [agent.settings.gamma ** k * item for k, item in enumerate(data.r)]
            cum_reward_list.append(np.sum(cum_reward))
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
        with open(os.path.join(agent_path, f"agent_update_{replay_idx}", "policy_gradients.pkl"), "wb") as f:
            pickle.dump(policy_gradient, f)
        with open(os.path.join(agent_path, f"agent_update_{replay_idx}", "policy_hessian.pkl"), "wb") as f:
            pickle.dump(policy_hessian, f)

        agent.performance_data.to_csv(os.path.join(agent_path, "performance_data.csv"))

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
    n_episodes_per_replay = 200

    parameterization_list = ["low", "medium", "high"]

    penalty_weights = 1e2


    
    actor_learning_rate = 1e-1
    trust_region_radius = actor_learning_rate
    use_momentum = True
    beta = 0.75
    beta_2 = 0.999
    eta = 0.9
    gamma = 0.99

    rl_settings = {
            "gamma": gamma,
            "actor_learning_rate": actor_learning_rate,
            "trust_region_radius": trust_region_radius,
            "scale_tr_radius_to_dimension": False,
            "adaptive_trust_region": True,
            "use_momentum": use_momentum,
            "momentum_beta": beta,
            "momentum_beta_2": beta_2,
            "momentum_eta": eta
        }

    for parameterization in parameterization_list:
        agent_path = os.path.join(
            "CSTR",
            "data",
            "dimensionality_investigation",
            f"gamma{gamma:.3f}, n_IC_per_replay{n_episodes_per_replay}",
            f"approx_newton_rad_{trust_region_radius:.1e}",
            f"{parameterization}_parameterization",
        )

        gc.collect()
                
        start_time = time.time()
        train(n_mpc_processes, penalty_weights, agent_path, rl_settings, n_episodes_per_replay, parameterization)
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