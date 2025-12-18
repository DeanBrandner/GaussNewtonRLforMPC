import os, time
import pandas as pd
import numpy as np

from itertools import product
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from torch.nn import ReLU, GELU

from environments import CSTR


def train(
        learning_rate: float = 1e-5,
        buffer_size: int = int(50e3),
        batch_size: int = 2048,
        tau: float = 1e-2,
        policy_delay: int = 10,
        max_steps_of_violation: int = 10):
    
    basepath = os.path.join("CSTR", "data", "sb3_td3")

    basepath = os.path.join(
        basepath,
        f"lr_{learning_rate:.2e}_bs_{buffer_size}_batch_{batch_size}_tau_{tau:.1e}_pd_{policy_delay}_msv_{max_steps_of_violation}"
    )

    terminate_on_cv = False
    CSTR.measurement_noise = False
    CSTR.additive_process_uncertainty = True
    CSTR.parametric_uncertainty = True

    CSTR.tight_initialization = False
    CSTR.terminal_cost_approximation = False
    CSTR.penalty_weight = 1e2

    CSTR.sb3_mode = True
    CSTR.sb3_n_ic = 200

    env = CSTR(seed = 20241128, terminate_on_cv = terminate_on_cv, max_steps_of_violation = max_steps_of_violation)
    env = Monitor(env= env, filename = os.path.join(basepath, "monitor.csv"))

    eval_env = CSTR(seed = 20250908, terminate_on_cv = terminate_on_cv, max_steps_of_violation = max_steps_of_violation)
    eval_env = Monitor(env= eval_env, filename = os.path.join(basepath, "eval_monitor.csv"))

    action_noise = NormalActionNoise(mean = np.zeros(env.action_space.shape), sigma = np.ones(env.action_space.shape) * 1e-2)

    checkpoint_callback = CheckpointCallback(
        save_freq = 50000,
        save_path = os.path.join(basepath, "logs", "cp_model"),
    )
    evaluation_callback = EvalCallback(
        eval_env = eval_env,
        best_model_save_path = os.path.join(basepath, "logs", "best_model_eval"),
        eval_freq = CSTR.sb3_n_ic * 105,
        n_eval_episodes = CSTR.sb3_n_ic,
        deterministic = True,
    )

    train_freq = (CSTR.sb3_n_ic, "episode")

    agent = TD3(
        policy = "MlpPolicy",
        env = env,
        learning_rate = learning_rate,
        buffer_size = buffer_size,
        # learning_starts = int(10e3),
        batch_size=batch_size,
        tau = tau,
        gamma = 0.99,
        train_freq = train_freq,
        gradient_steps = 500,
        policy_delay  = policy_delay,
        # action_noise=action_noise,
        policy_kwargs= {
            "activation_fn": GELU,
            "net_arch": {
                "pi": [64, 64],
                "qf": [256, 128],
                },
            },
        verbose = 2,
        seed = 99
    )

    start_time = time.time()
    agent.learn(
        total_timesteps=int(1e6), # This is the maximum amount of samples that the RL_MPC sees
        log_interval = 500,
        progress_bar=True,
        callback = [checkpoint_callback, evaluation_callback],
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    with open(os.path.join(basepath, "training_time.txt"), "w") as f:
        f.write(f"Training time: {training_time:.2f} seconds")

    agent.save(path = os.path.join(basepath, "final_agent"))

    monitored_data  = pd.read_csv(os.path.join(basepath, "monitor.csv"), skiprows=1)
    monitored_data_evaluation = pd.read_csv(os.path.join(basepath, "eval_monitor.csv"), skiprows=1)

    monitored_data_averaged = monitored_data.rolling(window = train_freq[0]).mean()
    monitored_data_evaluation_averaged = monitored_data_evaluation.rolling(window=train_freq[0]).mean()

    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (5, 4), constrained_layout = True)
    ax.plot(monitored_data_averaged["r"], label = "Training")
    ax.plot(monitored_data_evaluation_averaged["r"], label = "Validation")

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Episode Reward")
    ax.legend(loc = "upper right")

    ax.set_ylim([-250, 0])
    ax.grid()

    plt.savefig(os.path.join(basepath, "learning_curve.png"), dpi = 1200.0)
    return

if __name__ == "__main__":

    learning_rate_list = [1e-4, 1e-3]
    buffer_size_list = [int(50e3), int(100e3)]
    batch_size_list = [512, 1024]
    tau_list = [1e-1, 1e-2]
    policy_delay_list = [2, 5, 10]

    max_steps_of_violation = 1
    counter = 0
    for learning_rate, buffer_size, batch_size, tau, policy_delay in product(
            learning_rate_list,
            buffer_size_list,
            batch_size_list,
            tau_list,
            policy_delay_list
    ):
        train(
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            policy_delay=policy_delay,
            max_steps_of_violation=max_steps_of_violation
        )