import os, pickle, time
import numpy as np
import casadi as cd

from dataclasses import dataclass, field

from RL_MPC import RL_MPC
from helper import NLP_differentiator, tensor_vector_product, matrix_tensor_matrix_product, tensor_matrix_product

@dataclass
class Flags:
    differentiator_initialized: bool = False
    first_run: bool = True

@dataclass
class RL_settings:
    gamma: float = 1.
    actor_learning_rate: float = 1e-3
    adaptive_trust_region: bool = False
    trust_region_radius: float = 1e-2
    scale_tr_radius_to_dimension: bool = False
    exploration_noise: np.ndarray = 1e-6
    exploration_distribution: str = "normal"
    exploration_seed: int = 1
    verbose: int = 1

    regularization: str = "pos_eigen"
    clip_q_gradients: bool = False
    clip_jac_policy: bool = False
    use_momentum: bool = False
    use_adam: bool = False
    momentum_beta: float = 0.75
    momentum_beta_2: float = 0.999
    momentum_eta: float = 0.9
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.999
    adam_epsilon: float = 1e-8
    omegainv: float = 10.0
    D_history_length: int = 5
    use_scaled_actions: bool = False

@dataclass()
class Performance_data:
    episode: list = field(default_factory=list)
    n_samples: list = field(default_factory=list)
    time_replay: list = field(default_factory=list)

    def __init__(self):
        super().__init__()
        self.episode = [0]
        self.n_samples = []
        self.time_replay = []

    def update(self, agent):
        self.episode.append(self.episode[-1] + 1)
        self.n_samples.append(agent.observed_states.shape)
        self.time_replay.append(agent._time_replay)
        return

    def to_csv(self, path: str):
        
        episode_to_save = self.episode[:-1]
        n_samples_to_save = self.n_samples
        time_replay_to_save = self.time_replay

        from pandas import DataFrame
        df = DataFrame({
            "episode": episode_to_save,
            "n_samples": n_samples_to_save,
            "time_replay": time_replay_to_save
        })
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        df.to_csv(path, index = False)

    @classmethod
    def from_csv(cls, path: str):
        from pandas import read_csv
        df = read_csv(path)
        instance = cls()
        instance.episode = df["episode"].tolist()
        instance.episode.append(instance.episode[-1] + 1)  # Add one more episode for the next update
        instance.n_samples = df["n_samples"].tolist()
        instance.time_replay = df["time_replay"].tolist()
        return instance

class RL_MPC_agent():
    """
    This is an abstract class for general policy gradient MPC agents.
    """
    def __init__(
        self,
        mpc: RL_MPC = None,
        settings_dict: dict = {},
        **kwargs
        ):

        self.mpc = mpc
        self.action_shape = mpc.model._u.shape
        self.action_scale = (self.mpc._u_ub.master - self.mpc._u_lb.master).full().T

        # NOTE: The differentiator must be initialied in the child class
        self.differentiator = None


        self.flags = Flags()
        self.settings = RL_settings(**settings_dict)

        self.Q_func_transition_memory = []
        self.episodes_for_action_value_function = []

        self.performance_data = Performance_data()
    

    def _update_parameters(self, update: np.ndarray):
        p_template = self.mpc.get_p_template(1)
        p_template.master = self.mpc.p_fun(0)["_p", 0] + update

        self.mpc.set_p_fun(lambda t_now: p_template)
    

    def act(self, state: np.ndarray, old_action: np.ndarray = None, training: bool = False,):
        action = self.mpc.make_step(state, old_action)

        if not training:
            return action
        
        raise NotImplementedError("This function can only be used if training = False. The data, required for training changes from method to method and must be implemented in a child class.")

    def replay(self):
        raise NotImplementedError("This is an abstract method. It must be implemented in a child class.")

    def remember_transition_for_Q_func(
            self,
            state: np.ndarray,
            taken_action: np.ndarray,
            jac_action_prev_state:np.ndarray,
            jac_action_parameters:np.ndarray,
            reward: float,
            grad_reward_state: np.ndarray,
            grad_reward_action: np.ndarray,
            next_state: np.ndarray,
            jac_next_state_previous_state: np.ndarray,
            jac_next_state_taken_action: np.ndarray,
            termination: bool,
            truncation: bool
            ):
        self.Q_func_transition_memory.append(
            (
                state,
                taken_action,
                jac_action_prev_state,
                jac_action_parameters,
                reward,
                grad_reward_state,
                grad_reward_action,
                next_state,
                jac_next_state_previous_state,
                jac_next_state_taken_action,
                termination,
                truncation,
            )
        )

    def remember_episode_for_Q_func(self):
        self.episodes_for_action_value_function.append(self.Q_func_transition_memory)
        self.Q_func_transition_memory = []
            
    def _scale_actions(self):
        self.observed_previous_actions = (self.observed_previous_actions - self.mpc._u_lb.master.T.full()) / self.action_scale
        self.observed_taken_actions = (self.observed_taken_actions - self.mpc._u_lb.master.T.full()) / self.action_scale
        self.observed_jac_taken_action_parameters = self.observed_jac_taken_action_parameters / self.action_scale.reshape(1, self.action_scale.shape[1], 1)

        self.explored_previous_actions = (self.explored_previous_actions - self.mpc._u_lb.master.T.full()) / self.action_scale
        self.explored_taken_actions = (self.explored_taken_actions - self.mpc._u_lb.master.T.full()) / self.action_scale
        return

    def _compute_grad_V_theta(self, jac_action_parameters: list[np.ndarray], d_Q_d_a: list[np.ndarray]) -> np.ndarray:
        """
        This function computes the gradient of the state-value function with respect to the policy parameters along a full episode.
        """
        grad_V_theta = np.zeros((jac_action_parameters[0].T @ d_Q_d_a[0]).shape)
        for idx, (item_jac_action_params, item_dQ_da) in enumerate(zip(jac_action_parameters, d_Q_d_a)):
            grad_V_theta += self.settings.gamma ** idx * item_jac_action_params.T @ item_dQ_da
        return grad_V_theta
    
    def _compute_hess_V_theta_Gauss_newton(self, jac_action_parameters_per_episode, hess_Q_a_per_episode):
        
        H2 = np.zeros((jac_action_parameters_per_episode[0].T @ hess_Q_a_per_episode[0] @ jac_action_parameters_per_episode[0]).shape)

        for idx, (item_jac_action_params, item_hess_Q_a) in enumerate(zip(jac_action_parameters_per_episode, hess_Q_a_per_episode)):
            H2 += self.settings.gamma ** idx * (item_jac_action_params.T @ item_hess_Q_a @ item_jac_action_params)

        hess_V_theta = H2
        return hess_V_theta

    def _compute_hess_V_theta_approx_newton(self, jac_action_parameters_per_episode, jac_jac_action_parameters_per_episode, grad_Q_a_per_episode, hess_Q_a_per_episode):
        
        H1 = np.zeros((tensor_vector_product(jac_jac_action_parameters_per_episode[0], grad_Q_a_per_episode[0])).shape)
        H2 = np.zeros((H1.shape))

        for idx, (item_jac_jac_action_params, item_grad_Q_a) in enumerate(zip(jac_jac_action_parameters_per_episode, grad_Q_a_per_episode)):
            H1 += self.settings.gamma ** idx * tensor_vector_product(item_jac_jac_action_params, item_grad_Q_a)

        for idx, (item_jac_action_params, item_hess_Q_a) in enumerate(zip(jac_action_parameters_per_episode, hess_Q_a_per_episode)):
            H2 += self.settings.gamma ** idx * (item_jac_action_params.T @ item_hess_Q_a @ item_jac_action_params)

        hess_V_theta = H1 + H2
        return hess_V_theta
    
    def _regularize_policy_hessian(self, policy_hessian: np.ndarray, jac_action_parameters: np.ndarray):
        if self.settings.regularization.lower() == "fisher":
            reg_matrix = np.transpose(jac_action_parameters, axes = [0, 2, 1]) @ jac_action_parameters
            reg_matrix = np.mean(reg_matrix, axis = 0)
        elif self.settings.regularization.lower() == "identity":
            reg_matrix = np.eye(policy_hessian.shape[0])
        elif self.settings.regularization.lower() == "pos_eigen":
            eigenvalues, eigenvectors = np.linalg.eigh(policy_hessian)
            eigenvalues = -np.abs(eigenvalues) # Ensure eigenvalues are negative
            eigenvalues = np.clip(eigenvalues, a_max = -1e-8, a_min = -1e8)  # Ensure that the Hessian does not become indefinite.
            policy_hessian = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            reg_matrix = np.eye(policy_hessian.shape[0])
        else:
            raise ValueError(f"The regularization method {self.settings.regularization} is not supported. Please choose one of the following options: Fisher, Identity, pos_eigen")

        eigenvalues, eigenvectors = np.linalg.eigh(policy_hessian)
        reg_policy_hessian = policy_hessian.copy()
        rho = 1e-8
        while eigenvalues.max() > 0:
            reg_policy_hessian = policy_hessian  - rho * reg_matrix
            eigenvalues, eigenvectors = np.linalg.eigh(reg_policy_hessian)

            rho *= 10

        policy_hessian = reg_policy_hessian.copy()

        print(f"Eigenvalues of the policy hessian: {eigenvalues}")
        return policy_hessian



    # Loading and saving utilities! 
    def save(self, path: str, parameters_only: bool = False):
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.save_rl_parameters(path)

        if parameters_only:
            return
        
        attributes = self.__dict__.copy()

        mpc = attributes.pop("mpc")
        mpc.save(os.path.join(path, "mpc.pkl"))

        if "differentiator" in attributes:
            NLP_differentiator = attributes.pop("differentiator")
            with open(os.path.join(path, "differentiator.pkl"), "wb") as f:
                pickle.dump(NLP_differentiator, f)

        attributes.update({"class": self.__class__})
        
        with open(os.path.join(path, "agent.pkl"), "wb") as f:
            pickle.dump(attributes, f)
        return
    
    def save_rl_parameters(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "rl_params.pkl"), "wb") as f:
            pickle.dump(self.mpc.p_fun(0), f)
    
    def load_rl_parameters(self, path: str):
        with open(os.path.join(path, "rl_params.pkl"), "rb") as f:
            rl_params = pickle.load(f)

        p_template = self.mpc.get_p_template(1)
        p_template.master = rl_params["_p", 0]
        self.mpc.set_p_fun(lambda t_now: p_template)
        return self.mpc.p_fun(0)


    @staticmethod
    def load(path: str, load_differentiator: bool = True):
        with open(os.path.join(path, "agent.pkl"), "rb") as f:
            agent_attributes = pickle.load(f)

        cls = agent_attributes.pop("class")

        mpc = RL_MPC.load(os.path.join(path, "mpc.pkl"))

        rl_settings = agent_attributes.pop("settings")

        agent = cls(mpc, rl_settings.__dict__, init_differentiator = False)
        if load_differentiator:
            with open(os.path.join(path, "differentiator.pkl"), "rb") as f:
                agent.differentiator = pickle.load(f)
        
        for key, value in agent_attributes.items():
            setattr(agent, key, value)
        return agent

    def save_memories(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        memory = {
            "Q_func_transition_memory": self.Q_func_transition_memory,
        }
        with open(path, "wb") as f:
            pickle.dump(memory, f)

        self.Q_func_transition_memory = []
        return
        
    def synchronize_memories(self, path: str):
        files = os.listdir(path)

        for file in files:
            if file == "memory.pkl":
                continue

            with open(os.path.join(path, file), "rb") as f:
                memory = pickle.load(f)
            self.Q_func_transition_memory = memory["Q_func_transition_memory"]
            
            self.remember_episode_for_Q_func()

            os.remove(os.path.join(path, file))
        
        memory = {
            "episodes_for_action_value_function": self.episodes_for_action_value_function
        }
        with open(os.path.join(path, "memory.pkl"), "wb") as f:
            pickle.dump(memory, f)

        return 
    
class RL_MPC_GA_agent(RL_MPC_agent):

    def __init__(self, mpc: RL_MPC, settings_dict: dict = {}, init_differentiator: bool = True, **kwargs):
        super().__init__(mpc, settings_dict, **kwargs)

        if init_differentiator:
            self.differentiator_p = NLP_differentiator(self.mpc, ["_p"])
            self.differentiator_s = NLP_differentiator(self.mpc, ["_x0", "_u_prev"])
            self.flags.differentiator_initialized = True
        else:
            self.differentiator = None
            self.flags.differentiator_initialized = False

        self.update_counter = 0
        if self.settings.use_momentum:
            if self.settings.use_adam:
                raise ValueError("Momentum and Adam cannot be used together. Please choose one of them.")
            
            self.m = np.zeros((self.mpc.model._p.shape[0], 1))

        elif self.settings.use_adam:
            self.m = np.zeros((self.mpc.model._p.shape[0], 1))
            self.v = np.zeros((self.mpc.model._p.shape[0], 1))
            

    def act(self, state: np.ndarray, old_action: np.ndarray = None, training: bool = False):
        action = self.mpc.make_step(state, old_action)

        if not training:
            return action
        
        if not self.flags.differentiator_initialized:
            raise ValueError("The differentiator must be initialized before training.")
        if not self.mpc.solver_stats["success"]:
            print("\nThe solver did not converge. The action is not used for training.")
        
        if self.mpc.solver_stats["success"]:
            jac_action_parameters = self.differentiator_p.jac_action_parameters(self.mpc)
            jac_action_state = self.differentiator_s.jac_action_parameters(self.mpc)
        else:
            jac_action_parameters = cd.DM.zeros((self.mpc.model._u.shape[0], self.mpc.model._p.shape[0]))
            jac_action_state = cd.DM.zeros((self.mpc.model._u.shape[0], self.mpc.model._x.shape[0] + self.mpc.model._u.shape[0]))

        action_dict = {
            "action": action,
            "jac_action_parameters": jac_action_parameters,
            "jac_action_state": jac_action_state,
            "success": self.mpc.solver_stats["success"],
        }

        return action_dict
    
    def _compute_observed_clc(self):
        observed_clc = []

        for episode_idx, episode in enumerate(self.episodes_for_action_value_function):
            episode_clc = 0.0
            for trans_idx, transition in enumerate(reversed(episode)): 
                reward = transition[4]
                episode_clc = reward + self.settings.gamma * episode_clc
            
            observed_clc.append(episode_clc)
        
        observed_clc = np.stack(observed_clc).mean()

        return observed_clc
    
    def _prepare_calculations(self):

        ### Collect everything for gradient of the action-value function
        states = []

        taken_actions = []
        jacs_taken_action_parameters = []
        jacs_taken_action_state = []

        rewards = []
        grads_rewards_state = []
        grads_rewards_action = []
        d_reward_d_state = []

        next_states = []
        d_next_state_d_state = []
        d_next_state_d_action = []

        terminations = []
        truncations = []

        d_Q_d_a_list = []
        d_Q_d_s_list = []

        grad_V_theta_episode_list = []


        for episode in self.episodes_for_action_value_function:
            jac_action_parameters_per_episode = []
            grad_Q_a_per_episode = []
        

            for idx, (state, taken_action, jac_action_prev_state, jac_action_parameters, reward, grad_reward_state, grad_reward_action, next_state, jac_next_state_previous_state, jac_next_state_taken_action, termination, truncation) in enumerate(reversed(episode)):              
                states.append(state)

                taken_actions.append(taken_action)
                jacs_taken_action_parameters.append(jac_action_parameters)
                jacs_taken_action_state.append(jac_action_prev_state)

                rewards.append(reward)
                grads_rewards_state.append(grad_reward_state)
                grads_rewards_action.append(grad_reward_action)

                next_states.append(next_state)
                d_next_state_d_state.append(jac_next_state_previous_state)
                d_next_state_d_action.append(jac_next_state_taken_action)

                terminations.append(termination)
                truncations.append(truncation)

                d_r_d_s = grad_reward_state + jac_action_prev_state.T @ grad_reward_action
                d_s_next_d_s = jac_next_state_previous_state + jac_next_state_taken_action @ jac_action_prev_state
                if idx == 0:
                    d_Q_d_a = grad_reward_action.copy()
                    d_Q_d_s = d_r_d_s.copy()      
                else:
                    d_Q_d_a = grad_reward_action + self.settings.gamma * jac_next_state_taken_action.T @ d_Q_d_s
                    d_Q_d_s = d_r_d_s + self.settings.gamma * d_s_next_d_s.T @ d_Q_d_s

                d_Q_d_a_list.append(d_Q_d_a)
                d_Q_d_s_list.append(d_Q_d_s)

                jac_action_parameters_per_episode.append(jac_action_parameters)
                grad_Q_a_per_episode.append(d_Q_d_a)

            # Compute the gradient of the state-value function for this episode
            jac_action_parameters_per_episode.reverse()
            grad_Q_a_per_episode.reverse()
            grad_V_theta = self._compute_grad_V_theta(jac_action_parameters_per_episode, grad_Q_a_per_episode)
            grad_V_theta_episode_list.append(grad_V_theta)

        self.observed_states = np.hstack(states).T

        self.observed_taken_actions = np.hstack(taken_actions).T
        self.observed_jac_taken_action_parameters = np.stack(jacs_taken_action_parameters)
        self.observed_jac_taken_action_state = np.stack(jacs_taken_action_state)

        self.observed_rewards = np.array(rewards).reshape(-1, 1)
        self.observed_grad_rewards_state = np.stack(grads_rewards_state)
        self.observed_grad_rewards_action = np.stack(grads_rewards_action)

        self.observed_next_states = np.hstack(next_states).T
        self.observed_d_next_state_d_state = np.stack(d_next_state_d_state).T
        self.observed_d_next_state_d_action = np.stack(d_next_state_d_action).T

        self.observed_termination = np.array(terminations).reshape(-1, 1)
        self.observed_truncation = np.array(truncations).reshape(-1, 1)

        self.grad_Q_a = np.hstack(d_Q_d_a_list).T
        self.grad_Q_s = np.hstack(d_Q_d_s_list).T

        self.grad_V_theta = np.stack(grad_V_theta_episode_list)

        policy_gradient = self.grad_V_theta.mean(axis = 0)

        return policy_gradient
    
    @staticmethod
    def _compute_predicted_clc(local_clc: float, m_hat: np.ndarray, update: np.ndarray):
        predicted_clc = local_clc + m_hat.T @ update
        predicted_clc = predicted_clc[0, 0]
        return predicted_clc

    def replay(self):
        start_time_replay = time.time()

        observed_clc = self._compute_observed_clc()
        print(f"Observed CLC: {observed_clc:.4f}")
        
        policy_gradient = self._prepare_calculations()

        print(f"Policy gradient:")
        print(policy_gradient)
        
        update, m_hat, v_hat = self._compute_update(policy_gradient)

        print(f"Parameter update:")
        print(update)

        predicted_clc = RL_MPC_GA_agent._compute_predicted_clc(observed_clc, m_hat, update)
        print(f"Predicted CLC after update: {predicted_clc:.4f}")

        self._update_parameters(update)

        self.episodes_for_action_value_function = []

        end_time_replay = time.time()
        self._time_replay = end_time_replay - start_time_replay

        # Track performance metrics
        self.performance_data.update(self)

        return policy_gradient
    
    def _compute_update(self, policy_gradient: np.ndarray):
        """
        This function is used to compute the update for the parameters.
        """
        self.update_counter += 1
        if self.settings.use_momentum:
            # Update biased first moment estimate
            self.m = self.settings.adam_beta_1 * self.m + (1 - self.settings.adam_beta_1) * policy_gradient

            # Compute bias-corrected first moment estimate
            m_hat = self.m / (1 - self.settings.adam_beta_1 ** self.update_counter)

            update = m_hat * self.settings.actor_learning_rate

            v_hat = None

        elif self.settings.use_adam:
            # Update biased first moment estimate
            self.m = self.settings.adam_beta_1 * self.m + (1 - self.settings.adam_beta_1) * policy_gradient

            # Update biased second raw moment estimate
            self.v = self.settings.adam_beta_2 * self.v + (1 - self.settings.adam_beta_2) * (policy_gradient ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m / (1 - self.settings.adam_beta_1 ** self.update_counter)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v / (1 - self.settings.adam_beta_2 ** self.update_counter)

            update = m_hat / (np.sqrt(v_hat) + self.settings.adam_epsilon)

            update *= self.settings.actor_learning_rate
        else:
            # If not using Adam or momentum, just use the policy gradient
            update = self.settings.actor_learning_rate * policy_gradient
            m_hat = policy_gradient
            v_hat = None

        return update, m_hat, v_hat

class RL_MPC_GN_agent(RL_MPC_GA_agent):

    def __init__(self, mpc: RL_MPC, settings_dict: dict = {}, init_differentiator: bool = True, **kwargs):
        super().__init__(mpc, settings_dict, init_differentiator = False, **kwargs)
        
        if init_differentiator:
            self.differentiator_p = NLP_differentiator(self.mpc, ["_p"], second_order = False,)
            self.differentiator_s = NLP_differentiator(self.mpc, ["_x0", "_u_prev"], second_order = True,)
            self.flags.differentiator_initialized = True

        # Initialize the GN estimate
        if self.settings.use_momentum:
            self.D = np.zeros((self.mpc.model._p.shape[0], self.mpc.model._p.shape[0]))
            self.v = np.zeros((self.mpc.model._p.shape[0], 1))

        elif self.settings.use_adam:
            raise ValueError("You try to use Adam for second order optimization. Adam already scaled the first order update so the gradient is theoretically already corrected.")
    
    def act(self, state: np.ndarray, old_action: np.ndarray,  training: bool = False):
        action = self.mpc.make_step(state, old_action)

        if not training:
            return action
        
        if not self.flags.differentiator_initialized:
            raise ValueError("The differentiator must be initialized before training.")
        if not self.mpc.solver_stats["success"]:
            print("\nThe solver did not converge. The action is not used for training.")
        
        if self.mpc.solver_stats["success"]:
            jac_action_parameters = self.differentiator_p.jac_action_parameters(self.mpc)
            jac_action_state, jac_jac_action_state = self.differentiator_s.jac_jac_action_parameters_parameters(self.mpc)
        else:
            jac_action_parameters = np.zeros((self.differentiator_p.n_u, self.differentiator_p.n_p))
            jac_action_state = np.zeros((self.differentiator_s.n_u, self.differentiator_s.n_p))
            jac_jac_action_state = np.zeros((self.differentiator_s.n_u, self.differentiator_s.n_p, self.differentiator_s.n_p))

        action_dict = {
            "action": action,
            "jac_action_parameters": jac_action_parameters,
            "jac_action_state": jac_action_state,
            "jac_jac_action_states": jac_jac_action_state,
            "success": self.mpc.solver_stats["success"],
        }

        return action_dict
    
    def remember_transition_for_Q_func(
            self,
            state: np.ndarray,
            taken_action: np.ndarray,
            jac_action_prev_state:np.ndarray,
            jac_action_parameters:np.ndarray,
            jac_jac_action_state:np.ndarray,
            reward: float,
            grad_reward_state: np.ndarray,
            grad_reward_action: np.ndarray,
            hess_reward_state: np.ndarray,
            hess_reward_action: np.ndarray,
            jac_jac_reward_state_action: np.ndarray,
            next_state: np.ndarray,
            jac_next_state_previous_state: np.ndarray,
            jac_next_state_taken_action: np.ndarray,
            jac_jac_next_state_previous_state: np.ndarray,
            jac_jac_next_state_taken_action: np.ndarray,
            jac_jac_next_state_state_action: np.ndarray,
            termination: bool,
            truncation: bool
            ):
        self.Q_func_transition_memory.append(
            (
                state,
                taken_action,
                jac_action_prev_state,
                jac_action_parameters,
                jac_jac_action_state,
                reward,
                grad_reward_state,
                grad_reward_action,
                hess_reward_state,
                hess_reward_action,
                jac_jac_reward_state_action,
                next_state,
                jac_next_state_previous_state,
                jac_next_state_taken_action,
                jac_jac_next_state_previous_state,
                jac_jac_next_state_taken_action,
                jac_jac_next_state_state_action,
                termination,
                truncation,
            )
        )

    def _prepare_calculations(self):

        ### Collect everything for gradient of the action-value function
        states = []

        taken_actions = []
        jacs_taken_action_parameters = []
        jacs_taken_action_state = []

        rewards = []
        grads_rewards_state = []
        grads_rewards_action = []
        d_reward_d_state = []

        next_states = []
        d_next_state_d_state = []
        d_next_state_d_action = []

        terminations = []
        truncations = []

        d_Q_d_a_list = []
        d_Q_d_s_list = []

        d2_Q_d_a2_list = []
        d2_Q_d_s2_list = []

        grad_V_theta_episode_list = []
        hess_V_theta_episode_list = []




        for episode in self.episodes_for_action_value_function:
            jac_action_parameters_per_episode = []
            grad_Q_a_per_episode = []
            hess_Q_a_per_episode = []

            for idx, (state, taken_action, jac_action_prev_state, jac_action_parameters, jac_jac_action_state, reward, grad_reward_state, grad_reward_action, hess_reward_state, hess_reward_action, jac_jac_reward_state_action, next_state, jac_next_state_previous_state, jac_next_state_taken_action, jac_jac_next_state_previous_state, jac_jac_next_state_taken_action, jac_jac_next_state_state_action, termination, truncation) in enumerate(reversed(episode)):              
                states.append(state)

                taken_actions.append(taken_action)
                jacs_taken_action_parameters.append(jac_action_parameters)
                jacs_taken_action_state.append(jac_action_prev_state)

                rewards.append(reward)
                grads_rewards_state.append(grad_reward_state)
                grads_rewards_action.append(grad_reward_action)

                next_states.append(next_state)
                d_next_state_d_state.append(jac_next_state_previous_state)
                d_next_state_d_action.append(jac_next_state_taken_action)

                terminations.append(termination)
                truncations.append(truncation)

                d_r_d_s = grad_reward_state + jac_action_prev_state.T @ grad_reward_action
                d_s_next_d_s = jac_next_state_previous_state + jac_next_state_taken_action @ jac_action_prev_state

                d2_r_d_s2 = hess_reward_state + jac_action_prev_state.T @ hess_reward_action @ jac_action_prev_state
                d2_r_d_s2 += jac_jac_reward_state_action @ jac_action_prev_state +(jac_jac_reward_state_action @ jac_action_prev_state).T
                d2_r_d_s2 += tensor_vector_product(jac_jac_action_state, grad_reward_action)

                d2_s_next_d_s2 = jac_jac_next_state_previous_state + matrix_tensor_matrix_product(jac_action_prev_state.T, jac_jac_next_state_taken_action, jac_action_prev_state)
                d2_s_next_d_s2 += tensor_matrix_product(jac_jac_action_state, jac_next_state_taken_action)
                d2_s_next_d_s2 += matrix_tensor_matrix_product(np.eye(jac_jac_next_state_state_action.shape[1]), jac_jac_next_state_state_action, jac_action_prev_state)
                d2_s_next_d_s2 += matrix_tensor_matrix_product(jac_action_prev_state.T, np.transpose(jac_jac_next_state_state_action, axes = [0, 2, 1]), np.eye(np.transpose(jac_jac_next_state_state_action, axes = [0, 2, 1]).shape[2]))

                if idx == 0:
                    d_Q_d_a = grad_reward_action.copy()
                    d2_Q_d_a2 = hess_reward_action.copy()

                    d_Q_d_s = d_r_d_s.copy()   
                    d2_Q_d_s2 = d2_r_d_s2.copy()   
                    
                    v_value = reward

                else:
                    d_Q_d_a = grad_reward_action + self.settings.gamma * jac_next_state_taken_action.T @ d_Q_d_s

                    d2_Q_d_a2 = hess_reward_action.copy()
                    d2_Q_d_a2 += self.settings.gamma * (jac_next_state_taken_action.T @ d2_Q_d_s2 @ jac_next_state_taken_action)
                    d2_Q_d_a2 += self.settings.gamma * tensor_vector_product(jac_jac_next_state_taken_action, d_Q_d_s)

                    d2_Q_d_s2 = d2_r_d_s2.copy() + self.settings.gamma * (d_s_next_d_s.T @ d2_Q_d_s2 @ d_s_next_d_s)
                    d2_Q_d_s2 += self.settings.gamma * tensor_vector_product(d2_s_next_d_s2, d_Q_d_s)

                    d_Q_d_s = d_r_d_s + self.settings.gamma * d_s_next_d_s.T @ d_Q_d_s

                    v_value = reward + self.settings.gamma * v_value

                d_Q_d_a_list.append(d_Q_d_a)
                d_Q_d_s_list.append(d_Q_d_s)

                d2_Q_d_a2_list.append(d2_Q_d_a2)
                d2_Q_d_s2_list.append(d2_Q_d_s2)

                jac_action_parameters_per_episode.append(jac_action_parameters)
                grad_Q_a_per_episode.append(d_Q_d_a)
                hess_Q_a_per_episode.append(d2_Q_d_a2)


            # Compute the gradient and hessian of the state-value function for this episode
            jac_action_parameters_per_episode.reverse()
            grad_Q_a_per_episode.reverse()
            hess_Q_a_per_episode.reverse()
            grad_V_theta = self._compute_grad_V_theta(jac_action_parameters_per_episode, grad_Q_a_per_episode)
            hess_V_theta = self._compute_hess_V_theta_Gauss_newton(jac_action_parameters_per_episode, hess_Q_a_per_episode)
            grad_V_theta_episode_list.append(grad_V_theta)
            hess_V_theta_episode_list.append(hess_V_theta)

        self.observed_states = np.hstack(states).T

        self.observed_taken_actions = np.hstack(taken_actions).T
        self.observed_jac_taken_action_parameters = np.stack(jacs_taken_action_parameters)
        self.observed_jac_taken_action_state = np.stack(jacs_taken_action_state)

        self.observed_rewards = np.array(rewards).reshape(-1, 1)
        self.observed_grad_rewards_state = np.stack(grads_rewards_state)
        self.observed_grad_rewards_action = np.stack(grads_rewards_action)

        self.observed_next_states = np.hstack(next_states).T
        self.observed_d_next_state_d_state = np.stack(d_next_state_d_state).T
        self.observed_d_next_state_d_action = np.stack(d_next_state_d_action).T

        self.observed_termination = np.array(terminations).reshape(-1, 1)
        self.observed_truncation = np.array(truncations).reshape(-1, 1)

        self.grad_Q_a = np.hstack(d_Q_d_a_list).T
        self.grad_Q_s = np.hstack(d_Q_d_s_list).T

        self.hess_Q_a = np.stack(d2_Q_d_a2_list)
        self.hess_Q_s = np.stack(d2_Q_d_s2_list)

        self.grad_V_theta = np.stack(grad_V_theta_episode_list)
        self.hess_V_theta = np.stack(hess_V_theta_episode_list)

        policy_gradient = self.grad_V_theta.mean(axis = 0)
        policy_hessian = self.hess_V_theta.mean(axis = 0)

        return policy_gradient, policy_hessian

    def _compute_observed_clc(self):
        observed_clc = []

        for episode_idx, episode in enumerate(self.episodes_for_action_value_function):
            episode_clc = 0.0
            for trans_idx, transition in enumerate(reversed(episode)): 
                reward = transition[5]
                episode_clc = reward + self.settings.gamma * episode_clc
            
            observed_clc.append(episode_clc)
        
        observed_clc = np.stack(observed_clc).mean()

        return observed_clc
    
    @staticmethod
    def _compute_predicted_clc(local_clc: float, policy_gradient: np.ndarray, policy_hessian: np.ndarray, update: np.ndarray):
        predicted_clc = local_clc + policy_gradient.T @ update + 0.5 * update.T @ policy_hessian @ update
        predicted_clc = predicted_clc[0, 0]
        return predicted_clc
    
    def replay(self):
        start_time_replay = time.time()

        observed_clc = self._compute_observed_clc()
        print(f"Observed CLC: {observed_clc:.4f}")

        policy_gradient, policy_hessian = self._prepare_calculations()
        policy_hessian = self._regularize_policy_hessian(policy_hessian, self.observed_jac_taken_action_parameters)

        print(f"Policy gradient:")
        print(policy_gradient)
        print(f"Policy hessian:")
        print(policy_hessian)

        update, m_hat, D_hat = self._compute_update(policy_gradient, policy_hessian)

        print(f"Parameter update:")
        print(update)

        predicted_clc = RL_MPC_GN_agent._compute_predicted_clc(observed_clc, m_hat, D_hat, update)
        print(f"Predicted CLC after update: {predicted_clc:.4f}")


        self._update_parameters(update)

        self.episodes_for_action_value_function = []

        end_time_replay = time.time()
        self._time_replay = end_time_replay - start_time_replay

        # Track performance metrics
        self.performance_data.update(self)

        return policy_gradient, policy_hessian

    def _compute_update(self, policy_gradient: np.ndarray, policy_hessian: np.ndarray):
        self.update_counter += 1

        m_hat = policy_gradient.copy()
        v_hat = policy_gradient.copy() ** 2 
        D_hat = policy_hessian.copy()

        if self.settings.use_momentum:
            self.m = self.settings.momentum_beta * self.m + (1 - self.settings.momentum_beta) * policy_gradient
            m_hat = self.m / (1 - self.settings.momentum_beta ** self.update_counter)
            where_numerically_zero = np.where(np.abs(m_hat) < self.settings.adam_epsilon)
            m_hat[where_numerically_zero] = 0.0

            self.v = self.settings.momentum_beta_2 * self.v + (1 - self.settings.momentum_beta_2) * (policy_gradient ** 2)
            v_hat = self.v / (1 - self.settings.momentum_beta_2 ** self.update_counter)

            if self.update_counter == 1:
                D_init = -np.diag(np.sqrt(v_hat).flatten())
                self.D = D_init.copy()

            self.D = self.settings.momentum_eta * self.D + (1 - self.settings.momentum_eta) * policy_hessian
            D_hat = self.D / (1 - self.settings.momentum_eta ** (self.update_counter + 1))



        update = np.linalg.solve(D_hat, - m_hat)
        update_norm = np.sqrt(update.T @ update)

        tr_radius = self.settings.trust_region_radius
        if self.settings.scale_tr_radius_to_dimension and not self.settings.adaptive_trust_region:
            tr_radius *= np.sqrt(policy_gradient.shape[0])
        elif self.settings.adaptive_trust_region and not self.settings.scale_tr_radius_to_dimension:
            adam_update = self.settings.actor_learning_rate * (m_hat / (np.sqrt(v_hat) + self.settings.adam_epsilon))
            tr_radius = np.sqrt(adam_update.T @ adam_update).flatten()[0]
        else:
            print("Warning: Both adaptive_trust_region and scale_tr_radius_to_dimension are set to True. Only one of these options should be enabled. Proceeding with adaptive_trust_region.")
        
        if update_norm > tr_radius:

            print(f"Trust-region radius: {tr_radius:.3e}")
        
            step = cd.SX.sym("step", policy_gradient.shape)
            obj = - (m_hat.T @ step + 0.5 * step.T @ D_hat @ step)
            constr = step.T @ step

            nlp_dict = {
                "x": step,
                "f": obj,
                "g": constr,
            }
            nlpsol_options = {
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
            }

            solver = cd.nlpsol("solver", "ipopt", nlp_dict, nlpsol_options)

            lam_g0 = -(-np.abs(np.linalg.eigh(D_hat)[0])).max() * 0.5
            sol = solver(x0 = update, lam_g0 = lam_g0, lbg = tr_radius ** 2, ubg = tr_radius ** 2)
            
            if solver.stats()["return_status"] == "Solve_Succeeded":
                print("Trust-region subproblem solved successfully.")
            else:
                print("Warning: Trust-region subproblem solver returned with status: ", solver.stats()["return_status"])
            update = sol["x"].full()

        return update, m_hat, D_hat
    
class RL_MPC_AN_agent(RL_MPC_GN_agent):

    def __init__(self, mpc: RL_MPC, settings_dict: dict = {}, init_differentiator: bool = True, **kwargs):
        super().__init__(mpc, settings_dict, False, **kwargs)

        if init_differentiator:
            self.differentiator_p = NLP_differentiator(self.mpc, ["_p"], second_order = True,)
            self.differentiator_s = NLP_differentiator(self.mpc, ["_x0", "_u_prev"], second_order = True,)
            self.flags.differentiator_initialized = True

        self.D = np.zeros((self.mpc.model._p.shape[0], self.mpc.model._p.shape[0]))

    def act(self, state: np.ndarray, old_action: np.ndarray,  training: bool = False):
        action = self.mpc.make_step(state, old_action)

        if not training:
            return action
        
        if not self.flags.differentiator_initialized:
            raise ValueError("The differentiator must be initialized before training.")
        if not self.mpc.solver_stats["success"]:
            print("\nThe solver did not converge. The action is not used for training.")
        
        if self.mpc.solver_stats["success"]:
            jac_action_parameters, jac_jac_action_parameters = self.differentiator_p.jac_jac_action_parameters_parameters(self.mpc)
            jac_action_state, jac_jac_action_state = self.differentiator_s.jac_jac_action_parameters_parameters(self.mpc)
        else:
            jac_action_parameters = np.zeros((self.differentiator_p.n_u, self.differentiator_p.n_p))
            jac_action_state = np.zeros((self.differentiator_s.n_u, self.differentiator_s.n_p))
            jac_jac_action_parameters = np.zeros((self.differentiator_p.n_u, self.differentiator_p.n_p, self.differentiator_p.n_p))
            jac_jac_action_state = np.zeros((self.differentiator_s.n_u, self.differentiator_s.n_p, self.differentiator_s.n_p))

        action_dict = {
            "action": action,
            "jac_action_parameters": jac_action_parameters,
            "jac_action_state": jac_action_state,
            "jac_jac_action_parameters": jac_jac_action_parameters,
            "jac_jac_action_states": jac_jac_action_state,
            "success": self.mpc.solver_stats["success"],
        }

        return action_dict
    
    def remember_transition_for_Q_func(
            self,
            state: np.ndarray,
            taken_action: np.ndarray,
            jac_action_prev_state:np.ndarray,
            jac_action_parameters:np.ndarray,
            jac_jac_action_parameters:np.ndarray,
            jac_jac_action_state:np.ndarray,
            reward: float,
            grad_reward_state: np.ndarray,
            grad_reward_action: np.ndarray,
            hess_reward_state: np.ndarray,
            hess_reward_action: np.ndarray,
            jac_jac_reward_state_action: np.ndarray,
            next_state: np.ndarray,
            jac_next_state_previous_state: np.ndarray,
            jac_next_state_taken_action: np.ndarray,
            jac_jac_next_state_previous_state: np.ndarray,
            jac_jac_next_state_taken_action: np.ndarray,
            jac_jac_next_state_state_action: np.ndarray,
            termination: bool,
            truncation: bool
            ):
        self.Q_func_transition_memory.append(
            (
                state,
                taken_action,
                jac_action_prev_state,
                jac_action_parameters,
                jac_jac_action_parameters,
                jac_jac_action_state,
                reward,
                grad_reward_state,
                grad_reward_action,
                hess_reward_state,
                hess_reward_action,
                jac_jac_reward_state_action,
                next_state,
                jac_next_state_previous_state,
                jac_next_state_taken_action,
                jac_jac_next_state_previous_state,
                jac_jac_next_state_taken_action,
                jac_jac_next_state_state_action,
                termination,
                truncation,
            )
        )
    
    def _compute_observed_clc(self):
        observed_clc = []

        for episode_idx, episode in enumerate(self.episodes_for_action_value_function):
            episode_clc = 0.0
            for trans_idx, transition in enumerate(reversed(episode)): 
                reward = transition[6]
                episode_clc = reward + self.settings.gamma * episode_clc
            
            observed_clc.append(episode_clc)
        
        observed_clc = np.stack(observed_clc).mean()

        return observed_clc
    
    def _prepare_calculations(self):

        ### Collect everything for gradient of the action-value function
        states = []

        taken_actions = []
        jacs_taken_action_parameters = []
        jacs_taken_action_state = []
        jacs_jac_taken_action_parameters = []

        rewards = []
        grads_rewards_state = []
        grads_rewards_action = []
        d_reward_d_state = []

        next_states = []
        d_next_state_d_state = []
        d_next_state_d_action = []

        terminations = []
        truncations = []

        d_Q_d_a_list = []
        d_Q_d_s_list = []

        d2_Q_d_a2_list = []
        d2_Q_d_s2_list = []

        jacs_taken_action_parameters_test = []
        jacs_jac_taken_action_parameters_test = []
        d_Q_d_a_list_test = []
        d2_Q_d_a2_list_test = []

        v_values_test = []

        grad_V_theta_episode_list = []
        hess_V_theta_episode_list = []




        for episode in self.episodes_for_action_value_function:
            jac_action_parameters_per_episode = []
            jac_jac_action_parameters_per_episode = []
            grad_Q_a_per_episode = []
            hess_Q_a_per_episode = []

            for idx, (state, taken_action, jac_action_prev_state, jac_action_parameters, jac_jac_action_parameters, jac_jac_action_state, reward, grad_reward_state, grad_reward_action, hess_reward_state, hess_reward_action, jac_jac_reward_state_action, next_state, jac_next_state_previous_state, jac_next_state_taken_action, jac_jac_next_state_previous_state, jac_jac_next_state_taken_action, jac_jac_next_state_state_action, termination, truncation) in enumerate(reversed(episode)):              
                states.append(state)

                taken_actions.append(taken_action)
                jacs_taken_action_parameters.append(jac_action_parameters)
                jacs_taken_action_state.append(jac_action_prev_state)

                rewards.append(reward)
                grads_rewards_state.append(grad_reward_state)
                grads_rewards_action.append(grad_reward_action)
                jacs_jac_taken_action_parameters.append(jac_jac_action_parameters)

                next_states.append(next_state)
                d_next_state_d_state.append(jac_next_state_previous_state)
                d_next_state_d_action.append(jac_next_state_taken_action)

                terminations.append(termination)
                truncations.append(truncation)

                d_r_d_s = grad_reward_state + jac_action_prev_state.T @ grad_reward_action
                d_s_next_d_s = jac_next_state_previous_state + jac_next_state_taken_action @ jac_action_prev_state

                d2_r_d_s2 = hess_reward_state + jac_action_prev_state.T @ hess_reward_action @ jac_action_prev_state
                d2_r_d_s2 += jac_jac_reward_state_action @ jac_action_prev_state +(jac_jac_reward_state_action @ jac_action_prev_state).T
                d2_r_d_s2 += tensor_vector_product(jac_jac_action_state, grad_reward_action)

                d2_s_next_d_s2 = jac_jac_next_state_previous_state + matrix_tensor_matrix_product(jac_action_prev_state.T, jac_jac_next_state_taken_action, jac_action_prev_state)
                d2_s_next_d_s2 += tensor_matrix_product(jac_jac_action_state, jac_next_state_taken_action)
                d2_s_next_d_s2 += matrix_tensor_matrix_product(np.eye(jac_jac_next_state_state_action.shape[1]), jac_jac_next_state_state_action, jac_action_prev_state)
                d2_s_next_d_s2 += matrix_tensor_matrix_product(jac_action_prev_state.T, np.transpose(jac_jac_next_state_state_action, axes = [0, 2, 1]), np.eye(np.transpose(jac_jac_next_state_state_action, axes = [0, 2, 1]).shape[2]))

                if idx == 0:
                    d_Q_d_a = grad_reward_action.copy()
                    d2_Q_d_a2 = hess_reward_action.copy()

                    d_Q_d_s = d_r_d_s.copy()   
                    d2_Q_d_s2 = d2_r_d_s2.copy()   
                    
                    v_value = reward

                else:
                    d_Q_d_a = grad_reward_action + self.settings.gamma * jac_next_state_taken_action.T @ d_Q_d_s

                    d2_Q_d_a2 = hess_reward_action.copy()
                    d2_Q_d_a2 += self.settings.gamma * (jac_next_state_taken_action.T @ d2_Q_d_s2 @ jac_next_state_taken_action)
                    d2_Q_d_a2 += self.settings.gamma * tensor_vector_product(jac_jac_next_state_taken_action, d_Q_d_s)

                    d2_Q_d_s2 = d2_r_d_s2.copy() + self.settings.gamma * (d_s_next_d_s.T @ d2_Q_d_s2 @ d_s_next_d_s)
                    d2_Q_d_s2 += self.settings.gamma * tensor_vector_product(d2_s_next_d_s2, d_Q_d_s)

                    d_Q_d_s = d_r_d_s + self.settings.gamma * d_s_next_d_s.T @ d_Q_d_s

                    v_value = reward + self.settings.gamma * v_value

                d_Q_d_a_list.append(d_Q_d_a)
                d_Q_d_s_list.append(d_Q_d_s)

                d2_Q_d_a2_list.append(d2_Q_d_a2)
                d2_Q_d_s2_list.append(d2_Q_d_s2)
                v_values_test.append(v_value)

                jac_action_parameters_per_episode.append(jac_action_parameters)
                jac_jac_action_parameters_per_episode.append(jac_jac_action_parameters)
                grad_Q_a_per_episode.append(d_Q_d_a)
                hess_Q_a_per_episode.append(d2_Q_d_a2)

            jacs_taken_action_parameters_test.append(jac_action_parameters)
            jacs_jac_taken_action_parameters_test.append(jac_jac_action_parameters)
            d_Q_d_a_list_test.append(d_Q_d_a)
            d2_Q_d_a2_list_test.append(d2_Q_d_a2)

            # Compute the gradient and hessian of the state-value function for this episode
            jac_action_parameters_per_episode.reverse()
            grad_Q_a_per_episode.reverse()
            jac_jac_action_parameters_per_episode.reverse()
            hess_Q_a_per_episode.reverse()
            grad_V_theta = self._compute_grad_V_theta(jac_action_parameters_per_episode, grad_Q_a_per_episode)
            hess_V_theta = self._compute_hess_V_theta_approx_newton(jac_action_parameters_per_episode, jac_jac_action_parameters_per_episode, grad_Q_a_per_episode, hess_Q_a_per_episode)
            grad_V_theta_episode_list.append(grad_V_theta)
            hess_V_theta_episode_list.append(hess_V_theta)

        self.observed_states = np.hstack(states).T

        self.observed_taken_actions = np.hstack(taken_actions).T
        self.observed_jac_taken_action_parameters = np.stack(jacs_taken_action_parameters)
        self.observed_jac_taken_action_state = np.stack(jacs_taken_action_state)
        self.observed_jac_jac_taken_action_parameters = np.stack(jacs_jac_taken_action_parameters)

        self.observed_rewards = np.array(rewards).reshape(-1, 1)
        self.observed_grad_rewards_state = np.stack(grads_rewards_state)
        self.observed_grad_rewards_action = np.stack(grads_rewards_action)

        self.observed_next_states = np.hstack(next_states).T
        self.observed_d_next_state_d_state = np.stack(d_next_state_d_state).T
        self.observed_d_next_state_d_action = np.stack(d_next_state_d_action).T

        self.observed_termination = np.array(terminations).reshape(-1, 1)
        self.observed_truncation = np.array(truncations).reshape(-1, 1)

        self.grad_Q_a = np.hstack(d_Q_d_a_list).T
        self.grad_Q_s = np.hstack(d_Q_d_s_list).T

        self.hess_Q_a = np.stack(d2_Q_d_a2_list)
        self.hess_Q_s = np.stack(d2_Q_d_s2_list)

        self.grad_Q_a_test = np.hstack(d_Q_d_a_list_test).T
        self.hess_Q_a_test = np.stack(d2_Q_d_a2_list_test)

        self.observed_jac_taken_action_parameters_test = np.stack(jacs_taken_action_parameters_test)
        self.observed_jac_jac_taken_action_parameters_test = np.stack(jacs_jac_taken_action_parameters_test)

        self.v_values_test = np.array(v_values_test).reshape(-1, 1)

        self.grad_V_theta = np.stack(grad_V_theta_episode_list)
        self.hess_V_theta = np.stack(hess_V_theta_episode_list)

        policy_gradient = self.grad_V_theta.mean(axis = 0)
        policy_hessian = self.hess_V_theta.mean(axis = 0)

        return policy_gradient, policy_hessian
    
    def replay(self):
        start_time_replay = time.time()

        observed_clc = self._compute_observed_clc()
        print(f"Observed CLC: {observed_clc:.4f}")

        policy_gradient, policy_hessian = self._prepare_calculations()
        policy_hessian = self._regularize_policy_hessian(policy_hessian, self.observed_jac_taken_action_parameters)

        print(f"Policy gradient:")
        print(policy_gradient)
        print(f"Policy hessian:")
        print(policy_hessian)

        update, m_hat, D_hat = self._compute_update(policy_gradient, policy_hessian)
        print(f"Parameter update:")
        print(update)

        predicted_clc = RL_MPC_AN_agent._compute_predicted_clc(observed_clc, m_hat, D_hat, update)
        print(f"Predicted CLC after update: {predicted_clc:.4f}")

        self._update_parameters(update)
        

        self.episodes_for_action_value_function = []

        end_time_replay = time.time()
        self._time_replay = end_time_replay - start_time_replay

        # Track performance metrics
        self.performance_data.update(self)

        return policy_gradient, policy_hessian