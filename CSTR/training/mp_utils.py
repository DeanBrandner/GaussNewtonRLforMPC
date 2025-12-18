from multiprocessing import Queue
import casadi as cd
import os


def run_episode_loop(
        prepare_mpc,
        penalty_weight: float,
        Agent_class,
        rl_settings,
        Environment,
        agent_path: str,
        task_queue: Queue,
        environment_queue: Queue,
        progress_counter):
    
    mpc = prepare_mpc(penalty_weight)
    agent = Agent_class(mpc, rl_settings, prepare_v_func = False, prepare_a_func = False, prepare_q_func = False)
    
    while True:
        task = task_queue.get()
        if task == "STOP":
            break

        seed = task["seed"]
        if "parameters" in task:
            parameters = task["parameters"]
            p_template = agent.mpc.get_p_template(1)
            p_template.master = parameters
            agent.mpc.set_p_fun(lambda t_now: p_template)
            pass
        
        if "max_steps_of_violation" in task:
            max_steps_of_violation = task["max_steps_of_violation"]
        else:
            max_steps_of_violation = 5

        if "training" in task:
            training = task["training"]
        else:
            training = False

        if "testing" in task:
            testing = task["testing"]
        else:
            testing = False

        if "tight_initialization" in task:
            Environment.tight_initialization = task["tight_initialization"]
        if "terminal_cost_approximation" in task:
            Environment.terminal_cost_approximation = task["terminal_cost_approximation"]
        
        Environment.penalty_weight = penalty_weight

        if "measurement_noise" in task:
            Environment.measurement_noise = task["measurement_noise"]
        if "additive_process_uncertainty" in task:
            Environment.additive_process_uncertainty = task["additive_process_uncertainty"]
        if "parametric_uncertainty" in task:
            Environment.parametric_uncertainty = task["parametric_uncertainty"]

        if "diff_twice" in task:
            Environment.diff_twice = task["diff_twice"]

        run_episode(
            seed,
            agent,
            max_steps_of_violation,
            Environment,
            agent_path,
            training,
            testing,
            environment_queue,
            progress_counter
            )
        

def run_episode(seed:int, agent, max_steps_of_violation, Environment, agent_path:str, training:bool, testing:bool, environment_queue: Queue, progress_counter):

    terminate_on_cv = False
    if not testing:
        acting_environment = Environment(seed = 20241128 + seed, terminate_on_cv=terminate_on_cv, max_steps_of_violation = max_steps_of_violation, gamma = agent.settings.gamma, max_steps = 100)
    else:
        # For testing, we use a different seed to avoid conflicts with training
        acting_environment = Environment(seed = 20250614 + seed, terminate_on_cv=terminate_on_cv, max_steps_of_violation = max_steps_of_violation, gamma = agent.settings.gamma,)

    observation, info = acting_environment.reset(scale_observation = False)
    x0 = observation[0, :4].T.reshape(-1, 1).copy()
    u_prev = observation[0, 4:].T.reshape(-1, 1).copy()


    agent.mpc.x0.master = cd.DM(x0)
    agent.mpc.u0.master = cd.DM(u_prev)
    agent.mpc.set_initial_guess()

    

    termination, truncation = False, False

    counter = 0
    while not (termination or truncation):
        if training:
            action_dict = agent.act(state = x0, old_action = u_prev, training = True)
            action = action_dict["action"]
            jac_action_parameter = action_dict["jac_action_parameters"]
            jac_action_state = action_dict["jac_action_state"]
            if "jac_jac_action_parameters" in action_dict:
                jac_jac_action_parameter =  action_dict["jac_jac_action_parameters"]
            if "jac_jac_action_states" in action_dict:
                jac_jac_action_state = action_dict["jac_jac_action_states"]

            env_results = acting_environment.step(action = action, scaled_action = False, scale_observation = False)

            if not acting_environment.diff_twice:
                next_observation, jac_next_observation_s, jac_next_observation_a, reward, grad_reward_observation, grad_reward_action, termination, truncation, info = env_results
            else:
                next_observation, jac_next_observation_s, jac_next_observation_a, jac_jac_next_observation_s_s, jac_jac_next_observation_s_a, jac_jac_next_observation_a_a, reward, grad_reward_observation, grad_reward_action, hess_reward_observation, jac_jac_reward_sa, hess_reward_action, termination, truncation, info = env_results

            x_next = next_observation[0, :4].T.reshape(-1, 1).copy()
            u_prev_next = next_observation[0, 4:].T.reshape(-1, 1).copy()

            if not "jac_jac_action_parameters" in action_dict and not "jac_jac_action_states" in action_dict:
                # Adam scenario
                agent.remember_transition_for_Q_func(
                    state = observation.T,
                    taken_action = action,
                    jac_action_prev_state = jac_action_state,
                    jac_action_parameters = jac_action_parameter,
                    reward = reward,
                    grad_reward_state = grad_reward_observation,
                    grad_reward_action = grad_reward_action,
                    next_state = next_observation.T,
                    jac_next_state_previous_state = jac_next_observation_s,
                    jac_next_state_taken_action = jac_next_observation_a,
                    termination = termination,
                    truncation = truncation
                    )
            elif not "jac_jac_action_parameters" in action_dict and "jac_jac_action_states" in action_dict:
                # Gauss Newton scenario
                agent.remember_transition_for_Q_func(
                    state = observation.T,
                    taken_action = action,
                    jac_action_prev_state = jac_action_state,
                    jac_action_parameters = jac_action_parameter,
                    jac_jac_action_state = jac_jac_action_state,
                    reward = reward,
                    grad_reward_state = grad_reward_observation,
                    grad_reward_action = grad_reward_action,
                    hess_reward_state = hess_reward_observation,
                    hess_reward_action = hess_reward_action,
                    jac_jac_reward_state_action = jac_jac_reward_sa,
                    next_state = x_next,
                    jac_next_state_previous_state = jac_next_observation_s,
                    jac_next_state_taken_action = jac_next_observation_a,
                    jac_jac_next_state_previous_state = jac_jac_next_observation_s_s,
                    jac_jac_next_state_state_action = jac_jac_next_observation_s_a,
                    jac_jac_next_state_taken_action = jac_jac_next_observation_a_a,
                    termination = termination,
                    truncation = truncation
                )
            else:
                # Approx Newton scenario
                agent.remember_transition_for_Q_func(
                    state = observation.T,
                    taken_action = action,
                    jac_action_prev_state = jac_action_state,
                    jac_action_parameters = jac_action_parameter,
                    jac_jac_action_parameters = jac_jac_action_parameter,
                    jac_jac_action_state = jac_jac_action_state,
                    reward = reward,
                    grad_reward_state = grad_reward_observation,
                    grad_reward_action = grad_reward_action,
                    hess_reward_state = hess_reward_observation,
                    hess_reward_action = hess_reward_action,
                    jac_jac_reward_state_action = jac_jac_reward_sa,
                    next_state = x_next,
                    jac_next_state_previous_state = jac_next_observation_s,
                    jac_next_state_taken_action = jac_next_observation_a,
                    jac_jac_next_state_previous_state = jac_jac_next_observation_s_s,
                    jac_jac_next_state_state_action = jac_jac_next_observation_s_a,
                    jac_jac_next_state_taken_action = jac_jac_next_observation_a_a,
                    termination = termination,
                    truncation = truncation
                )

            observation = next_observation.copy()

            x0 = next_observation[0, :4].T.reshape(-1, 1).copy()
            u_prev = next_observation[0, 4:].T.reshape(-1, 1).copy()

        else:
            action = agent.act(state = x0, old_action = u_prev, training = False)
            jac_action_state = 0
            jac_action_parameter = 0

            env_results = acting_environment.step(action = action, scaled_action = False, scale_observation = False)

            next_observation, jac_next_observation_s, jac_next_observation_a, reward, grad_reward_observation, grad_reward_action, termination, truncation, info = env_results
            x_next = next_observation[0, :4].T.reshape(-1, 1).copy()
            u_prev_next = next_observation[0, 4:].T.reshape(-1, 1).copy()

            agent.remember_transition_for_Q_func(
                state = observation.T,
                taken_action = action,
                jac_action_prev_state = jac_action_state,
                jac_action_parameters = jac_action_parameter,
                reward = reward,
                grad_reward_state = grad_reward_observation,
                grad_reward_action = grad_reward_action,
                next_state = next_observation.T,
                jac_next_state_previous_state = jac_next_observation_s,
                jac_next_state_taken_action = jac_next_observation_a,
                termination = termination,
                truncation = truncation
                )
                
            observation = next_observation.copy()

            x0 = next_observation[0, :4].T.reshape(-1, 1).copy()
            u_prev = next_observation[0, 4:].T.reshape(-1, 1).copy()
    

    agent.save_memories(os.path.join(agent_path, "memories", f"replay{seed}.pkl"))
    agent.mpc.reset_history()

    acting_environment.data.compactify()

    environment_queue.put(
        {
            "episode": 0,
            "seed": seed,
            "data": acting_environment.data,
            "termination": termination,
            "truncation": truncation,
        }
    )

    with progress_counter.get_lock():
        progress_counter.value += 1

    return