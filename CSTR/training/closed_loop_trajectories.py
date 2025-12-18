import os, pickle
from tqdm import tqdm

# Import necessary modules and functions
from environments import CSTR as Environment
from rl_mpc_agents import RL_MPC_agent as Agent




def closed_loop_control(
        agent: Agent,
        closed_loop_results_path: str,
        n_clc: int,
        gamma: float = 0.99,
        ):
    
    env = Environment(
        seed = 99999,
        terminate_on_cv=False,
        max_steps = 101,
        gamma = gamma)
    
    env.additive_process_uncertainty = True
    env.parametric_uncertainty = True
    env.penalty_weight = 1e2

    if not os.path.exists(closed_loop_results_path):
        os.makedirs(closed_loop_results_path)


    if isinstance(agent, Agent):
        for ic in tqdm(iterable=range(n_clc), desc = "Initial conditions"):
            observation, info = env.reset(seed = env.settings.seed + ic, scale_observation=False)
            agent.mpc.reset_history()

            for step in range(env.settings.max_steps):
                state = observation.T[:4]
                old_action = observation.T[4:]

                action = agent.act(state = state, old_action = old_action, training = False)
                env_results = env.step(action = action.T, scaled_action = False, scale_observation = False)            
                observation, _, _, reward, _, _, terminated, truncated, info = env_results
            
            env.data.compactify()

            with open(os.path.join(closed_loop_results_path, f"episode_data_ic_{ic}.pkl"), "wb") as f:
                pickle.dump(env.data, f)

    elif isinstance(agent, SB3Agent):
        env.sb3_mode = True
        env.sb3_test_mode = True
        for ic in tqdm(iterable=range(n_clc), desc = "Initial conditions"):
            observation, info = env.reset(seed = env.settings.seed + ic, scale_observation=True)

            for step in range(env.settings.max_steps):
                action, _states = agent.predict(observation, deterministic = True)
                env_results = env.step(action = action, scaled_action = True, scale_observation = True)            
                observation, reward, terminated, truncated, info = env_results

            env.data.compactify()

            with open(os.path.join(closed_loop_results_path, f"episode_data_ic_{ic}.pkl"), "wb") as f:
                pickle.dump(env.data, f)
    
    return

if __name__ == "__main__":

    n_initial_conditions = 200

    n_clc = 25

    gamma = 0.99
    learning_rate = 1e-2

    parameterization = "low"

    methods = ["Untrained", f"Adam_lr_{learning_rate:.1e}", f"approx_newton_rad_{10 * learning_rate:.1e}", f"gauss_newton_rad_{10 * learning_rate:.1e}", os.path.join("sb3_td3", "lr_1.00e-03_bs_100000_batch_1024_tau_1.0e-01_pd_10_msv_1")]

    agent_base_path = os.path.join("CSTR", "data", "dimensionality_investigation", f"gamma{gamma:.3f}, n_IC_per_replay{n_initial_conditions}")
    clc_base_path = os.path.join("CSTR", "data", "closed_loops", f"gamma{gamma:.3f}, n_IC_per_replay{n_initial_conditions}")

    for idx, method in enumerate(methods):
        if method == "Untrained":
            from mpc_collection import get_rl_mpc
            mpc = get_rl_mpc()
            agent = Agent(mpc, gamma = gamma)

            clc_path = os.path.join(clc_base_path, "untrained", f"{parameterization}_parameterization")
        elif "sb3" in method:
            agent_path = os.path.join("CSTR", "data", method, "logs", "best_model_eval", "best_model.zip")
            clc_path = os.path.join(clc_base_path, method)

            if "td3" in method:
                from stable_baselines3 import TD3 as SB3Agent
            agent = SB3Agent.load(agent_path)
        else:
            agent_path = os.path.join(agent_base_path, method, f"{parameterization}_parameterization")
            clc_path = os.path.join(clc_base_path, method, f"{parameterization}_parameterization")
        
            agent = Agent.load(os.path.join(agent_path, "agent_update"), load_differentiator = False)

        if not method == "Untrained" and not "sb3" in method:
            folders = [folder for folder in os.listdir(agent_path) if folder.startswith("agent_update_")]
            iters = [int(folder.split("_")[-1]) for folder in folders]
            final_iter = max(iters) - 1  # Use the second to last iteration to avoid incomplete training runs
            
            trained_agent_path = os.path.join(agent_path, f"agent_update_{final_iter}")
            clc_path = os.path.join(clc_path, f"agent_update_{final_iter}")
            
            agent.load_rl_parameters(trained_agent_path)

        if not os.path.exists(clc_path):
            os.makedirs(clc_path)

        # Run full parametric performance computation
        closed_loop_control(
            agent,
            clc_path,
            n_clc,
            gamma,
        )