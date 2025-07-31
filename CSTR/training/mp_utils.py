from multiprocessing import Queue
import casadi as cd
import os


def run_episode_loop(prepare_mpc, Agent_class, rl_settings, Environment, agent_path: str, task_queue: Queue, environment_queue: Queue, progress_counter):
    
    mpc = prepare_mpc()
    agent = Agent_class(mpc, rl_settings)
    
    while True:
        task = task_queue.get()
        if task == "STOP":
            break

        agent.exploration_noise.reset()

        seed = task["seed"]
        if "parameters" in task:
            parameters = task["parameters"]
            p_template = agent.mpc.get_p_template(1)
            p_template.master = parameters
            mpc.set_p_fun(lambda t_now: p_template)
        
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

    if not testing:
        acting_environment = Environment(seed = 20241128 + seed, terminate_on_cv=False, max_steps_of_violation = max_steps_of_violation)
    else:
        # For testing, we use a different seed to avoid conflicts with training
        acting_environment = Environment(seed = 20250614 + seed, terminate_on_cv=False, max_steps_of_violation = max_steps_of_violation)

    observation, info = acting_environment.reset(scale_observation = False)
    x0 = observation[0, :4].T.reshape(-1, 1).copy()
    u_prev = observation[0, 4:].T.reshape(-1, 1).copy()

    if training:
        exploration_environment = Environment(seed = 20241128 + seed, terminate_on_cv=False, max_steps_of_violation = max_steps_of_violation)
        _ = exploration_environment.set_observation(observation, scale_observation = False)
    

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
            if "jac_jac_action_parameters" in action_dict:
                jac_jac_action_parameter =  action_dict["jac_jac_action_parameters"]

            action_exp = agent.explore(action)

            _ = exploration_environment.set_observation(observation, scale_observation = False)

            next_observation, reward, termination, truncation, info = acting_environment.step(action = action, scaled_action = False, scale_observation = False)
            next_observation_exp, reward_exp, termination_exp, truncation_exp, info_exp = exploration_environment.step(action = action_exp, scaled_action = False, scale_observation = False)

            x_next = next_observation[0, :4].T.reshape(-1, 1).copy()
            u_prev_next = next_observation[0, 4:].T.reshape(-1, 1).copy()

            x_next_exp = next_observation_exp[0, :4].T.reshape(-1, 1).copy()
            u_prev_next_exp = next_observation_exp[0, 4:].T.reshape(-1, 1).copy()

            if not "jac_jac_action_parameters" in action_dict:
                agent.remember_transition_for_V_func(
                    state = x0,
                    previous_action = u_prev,
                    taken_action = action, 
                    jac_action_parameters = jac_action_parameter,
                    reward = reward,
                    next_state = x_next,
                    termination = termination,
                    truncation = truncation
                    )
            else:
                agent.remember_transition_for_V_func(
                    state = x0,
                    previous_action = u_prev,
                    taken_action = action, 
                    jac_action_parameters = jac_action_parameter,
                    jac_jac_action_parameters = jac_jac_action_parameter,
                    reward = reward,
                    next_state = x_next,
                    termination = termination,
                    truncation = truncation
                    )
            agent.remember_transition_for_Q_func(
                state = x0,
                previous_action = u_prev,
                taken_action = action_exp,
                reward = reward_exp,
                next_state = x_next_exp,
                termination = termination_exp,
                truncation = truncation_exp
                )

            observation = next_observation.copy()

            x0 = next_observation[0, :4].T.reshape(-1, 1).copy()
            u_prev = next_observation[0, 4:].T.reshape(-1, 1).copy()

        else:
            action = agent.act(state = x0, old_action = u_prev, training = False)
            jac_action_parameter = 0

            next_observation, reward, termination, truncation, info = acting_environment.step(action = action, scaled_action = False, scale_observation = False)

            x_next = next_observation[0, :4].T.reshape(-1, 1).copy()
            u_prev_next = next_observation[0, 4:].T.reshape(-1, 1).copy()

            agent.remember_transition_for_V_func(
                state = x0,
                previous_action = u_prev,
                taken_action = action, 
                jac_action_parameters = jac_action_parameter,
                reward = reward,
                next_state = x_next,
                termination = termination,
                truncation = truncation
                )  

            x0 = x_next.copy()
            u_prev = u_prev_next.copy()
    

    agent.save_memories(os.path.join(agent_path, "memories", f"replay{seed}.pkl"))

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