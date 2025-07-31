import numpy as np
import casadi as cd
import pickle
import os
from RL_MPC import RL_MPC
from helper import NLP_differentiator, tensor_vector_product, Noise_generator
from dataclasses import dataclass, field

from Q_func_model import Q_approximator

from keras.src.utils import set_random_seed
set_random_seed(1)

from keras.src.backend import set_floatx
set_floatx("float64")

from keras.src.models import Model as keras_model
from keras.src.layers import Dense, Input, Concatenate
from keras.src.saving import load_model as load_nn_model
from keras.src.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from sklearn.preprocessing import MinMaxScaler

from tensorflow import GradientTape, constant
from tensorflow.random import set_seed as tf_set_seed
tf_set_seed(1)

@dataclass
class Flags:
    differentiator_initialized: bool = False
    first_run: bool = True

@dataclass
class RL_settings:
    gamma: float = 1.
    actor_learning_rate: float = 1e-3
    exploration_noise: np.ndarray = 1e-6
    exploration_distribution: str = "normal"
    exploration_seed: int = 1
    verbose: int = 1

    q_func_architecture: list = field(default_factory= lambda: [64, 64])
    q_func_activation: str = "tanh"
    q_func_learning_rate: float = 1e-3
    q_func_batch_size: int = 64
    q_func_optimizer: str = "adam"
    q_func_loss: str = "mse"
    q_func_epochs: int = 10
    q_func_always_reset: bool = False

    regularization: str = "pos_eigen"
    clip_q_gradients: bool = False
    clip_jac_policy: bool = False
    use_momentum: bool = False
    use_adam: bool = False
    momentum_beta: float = 0.75
    momentum_eta: float = 0.9
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.999
    adam_epsilon: float = 1e-8
    omegainv: float = 10
    use_scaled_actions: bool = True


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

        self.exploration_noise = Noise_generator(shape = self.action_shape, noise_type = self.settings.exploration_distribution, noise_std = self.settings.exploration_noise, seed = self.settings.exploration_seed)

        self.V_func_transition_memory = []
        self.Q_func_transition_memory = []

        self.episodes_for_state_value_function = []
        self.episodes_for_action_value_function = []

        self.q_func = self._prepare_q_func()


    def _prepare_q_func(self):
        q_func_input_x = Input(shape = (self.mpc.model._x.shape[0],), name = "q_func_input_x")
        q_func_input_u_prev = Input(shape=(self.mpc.model._u.shape[0],), name = "q_func_input_u_prev")
        q_func_input_taken_action = Input(shape=(self.mpc.model._u.shape[0],), name = "q_func_input_taken_action")

        v_func_input = Concatenate(name = "stacked_input_for_v")([q_func_input_x, q_func_input_u_prev])
        a_func_input = Concatenate(name = "stacked_input_for_a")([q_func_input_x, q_func_input_u_prev, q_func_input_taken_action])

        specific_a_func_input_list = []
        for idx in range(q_func_input_taken_action.shape[1]):
            a_func_input_exploration_i = Concatenate(name = f"stacked_input_for_a_{idx}")([q_func_input_x, q_func_input_u_prev, q_func_input_taken_action[:, idx:idx+1]])
            specific_a_func_input_list.append(a_func_input_exploration_i)

        for idx, neurons in enumerate(self.settings.q_func_architecture):
            if idx == 0:
                v_next = Dense(neurons, activation = self.settings.q_func_activation)(v_func_input)
                a_next_list =[]
                for a_func_input in specific_a_func_input_list:
                    a_next = Dense(neurons, activation = self.settings.q_func_activation)(a_func_input)
                    a_next_list.append(a_next)
            else:
                v_next = Dense(neurons, activation = self.settings.q_func_activation)(v_next)
                for jdx, a_next in enumerate(a_next_list):
                    a_next = Dense(neurons, activation = self.settings.q_func_activation)(a_next)
                    a_next_list[jdx] = a_next

        v_func_output = Dense(1, activation = "linear", name = "v_func_output")(v_next)
        a_next = Concatenate(name = "stacked_a_next_output")(a_next_list)
        a_func_output = Dense(1, activation = "linear", name = "a_func_output")(a_next)

        output = Concatenate(name = "stacked_output")([v_func_output, a_func_output])

        q_func_model = Q_approximator(inputs = [q_func_input_x, q_func_input_u_prev, q_func_input_taken_action], outputs = output, name ="V_A_func")
        
        if self.settings.q_func_optimizer.lower() == "adam":
            from keras.src.optimizers import Adam as Optimizer
        else:
            raise NotImplementedError(f"You try to use {self.settings.q_func_optimizer} as an optimizer, which is not supported yet. Please choose one of the following options: Adam")
        
        q_func_model.compile(
            optimizer = Optimizer(learning_rate=self.settings.q_func_learning_rate),
            loss = self.settings.q_func_loss,
            metrics = ["mse"],
            # run_eagerly= True,
            )
        return q_func_model

    def _learn_q_function(
            self,
            observed_states,
            observed_previous_actions,
            observed_taken_actions,
            observed_rewards,
            observed_v_values,
            observed_next_states,
            observed_termination,
            explored_states,
            explored_previous_actions,
            explored_taken_actions,
            explored_rewards,
            explored_next_states,
            explored_termination,
            ):
        
        self.observed_states_scaler = MinMaxScaler()
        self.observed_previous_actions_scaler = MinMaxScaler()
        self.observed_v_values_scaler = MinMaxScaler()

        scaled_observed_states = self.observed_states_scaler.fit_transform(observed_states)
        scaled_observed_previous_actions = self.observed_previous_actions_scaler.fit_transform(observed_previous_actions)
        scaled_observed_v_values = self.observed_v_values_scaler.fit_transform(observed_v_values)

        scaled_explored_rewards = explored_rewards * self.observed_v_values_scaler.scale_
        scaled_observed_next_states = self.observed_states_scaler.transform(observed_next_states)
        scaled_observed_taken_actions = self.observed_previous_actions_scaler.transform(observed_taken_actions)
        scaled_explored_next_states = self.observed_states_scaler.transform(explored_next_states)
        scaled_explored_taken_actions = self.observed_previous_actions_scaler.transform(explored_taken_actions)

        observed_termination = np.array(observed_termination, dtype = np.float64)
        explored_termination = np.array(explored_termination, dtype = np.float64)

        V_min = np.ones((scaled_observed_next_states.shape[0], 1)) * self.observed_v_values_scaler.min_
        x_data = [scaled_observed_states, scaled_observed_previous_actions, scaled_observed_taken_actions, scaled_explored_taken_actions,  scaled_explored_next_states, explored_termination, V_min]
        y_data = [scaled_observed_v_values, scaled_explored_rewards]
        self.q_func.fit(
            x = x_data,
            y = y_data,
            batch_size = self.settings.q_func_batch_size,
            epochs = self.settings.q_func_epochs,
            verbose = 1,
            # validation_split = 0.0,
            callbacks = [
                ReduceLROnPlateau(monitor = "loss", factor = 0.1, patience = 30, min_lr = 1e-6, min_delta = 1e-7, start_from_epoch = 100),
                EarlyStopping(monitor = "loss", min_delta = 1e-7, patience = 50, verbose = 1, restore_best_weights = True, start_from_epoch = 100)
                ],
        )

        self.q_func.optimizer.learning_rate = self.settings.q_func_learning_rate
        return

    def _update_parameters(self, update: np.ndarray):
        p_template = self.mpc.get_p_template(1)
        p_template.master = self.mpc.p_fun(0)["_p", 0] + update

        self.mpc.set_p_fun(lambda t_now: p_template)
            
    def act(self, state: np.ndarray, old_action: np.ndarray = None, training: bool = False,):
        action = self.mpc.make_step(state, old_action)

        if not training:
            return action
        
        raise NotImplementedError("This function can only be used if training = False. The data, required for training changes from method to method and must be implemented in a child class.")

    def explore(self, action):
        applied_noise = self.exploration_noise()

        proposed_action = action.copy() + applied_noise

        vlb = proposed_action < self.mpc._u_lb.master.full()
        vub = proposed_action > self.mpc._u_ub.master.full()

        proposed_action[vlb] = action[vlb] - applied_noise[vlb]
        proposed_action[vub] = action[vub] - applied_noise[vub]
        return proposed_action
    
    def remember_transition_for_V_func(
            self,
            state: np.ndarray,
            previous_action: np.ndarray,
            taken_action: np.ndarray,
            jac_action_parameters:np.ndarray,
            reward: float,
            next_state: np.ndarray,
            termination: bool,
            truncation: bool
            ):
        self.V_func_transition_memory.append(
            (
                state,
                previous_action,
                taken_action,
                jac_action_parameters,
                reward,
                next_state,
                termination,
                truncation,
            )
        )

    def remember_episode_for_V_func(self):
        self.episodes_for_state_value_function.append(self.V_func_transition_memory)
        self.V_func_transition_memory = []
    
    def remember_transition_for_Q_func(
            self,
            state: np.ndarray,
            previous_action: np.ndarray,
            taken_action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            termination: bool,
            truncation: bool
            ):
        self.Q_func_transition_memory.append(
            (
                state,
                previous_action,
                taken_action,
                reward,
                next_state,
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

    @staticmethod
    def compute_policy_gradient(jac_action_parameters: np.ndarray, grad_Q_a: np.ndarray):
        if len(grad_Q_a.shape) == 2:
            grad_Q_a = np.expand_dims(grad_Q_a, axis = -1)
        
        # Compute the policy gradient. For this, we need jac_action_parameters and grad_Q_a at taken action
        policy_gradient = np.transpose(jac_action_parameters, axes = [0, 2, 1]) @ grad_Q_a
        policy_gradient = np.mean(policy_gradient, axis = 0)

        return policy_gradient

    @staticmethod
    def compute_Gauss_Newton_matrix(jac_action_parameters: np.ndarray, hess_Q_a: np.ndarray):
        # Compute the Gauss-Newton matrix. For this, we need jac_action_parameters and hess_Q_a at taken action
        Gauss_Newton_matrix = np.transpose(jac_action_parameters, axes = [0, 2, 1]) @ hess_Q_a @ jac_action_parameters
        Gauss_Newton_matrix = np.mean(Gauss_Newton_matrix, axis = 0)
        return Gauss_Newton_matrix
    
    @staticmethod
    def compute_approximate_Newton_matrix(jac_action_parameters: np.ndarray, jac_jac_action_parameters: np.ndarray, grad_Q_a: np.ndarray, hess_Q_a: np.ndarray):
        if len(grad_Q_a.shape) == 2:
            grad_Q_a = np.expand_dims(grad_Q_a, axis = -1)
        
        # Compute the policy gradient. For this, we need jac_action_parameters and grad_a_func at taken action
        policy_hessian = tensor_vector_product(jac_jac_action_parameters, grad_Q_a)
        policy_hessian += np.transpose(jac_action_parameters, axes = [0, 2, 1]) @ hess_Q_a @ jac_action_parameters
        policy_hessian = np.mean(policy_hessian, axis = 0)
        return policy_hessian
    
    
    def _regularize_policy_hessian(self, policy_hessian: np.ndarray, jac_action_parameters: np.ndarray):
        if self.settings.regularization.lower() == "fisher":
            reg_matrix = np.transpose(jac_action_parameters, axes = [0, 2, 1]) @ jac_action_parameters
            reg_matrix = np.mean(reg_matrix, axis = 0)
        elif self.settings.regularization.lower() == "identity":
            reg_matrix = np.eye(policy_hessian.shape[0])
        elif self.settings.regularization.lower() == "pos_eigen":
            eigenvalues, eigenvectors = np.linalg.eigh(policy_hessian)
            eigenvalues = -np.abs(eigenvalues) # Ensure eigenvalues are positive
            eigenvalues = np.clip(eigenvalues, a_max = -1e-3, a_min = None)  # Ensure that the Hessian does not become indefinite.
            policy_hessian = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        else:
            raise ValueError(f"The regularization method {self.settings.regularization} is not supported. Please choose one of the following options: Fisher, Identity, pos_eigen")

        eigenvalues, eigenvectors = np.linalg.eigh(policy_hessian)
        reg_policy_hessian = policy_hessian.copy()
        rho = 1e-12
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
        self.save_q_func(path)

        if parameters_only:
            return
        
        attributes = self.__dict__.copy()

        mpc = attributes.pop("mpc")
        mpc.save(os.path.join(path, "mpc.pkl"))

        if "differentiator" in attributes:
            NLP_differentiator = attributes.pop("differentiator")
            with open(os.path.join(path, "differentiator.pkl"), "wb") as f:
                pickle.dump(NLP_differentiator, f)

        q_func = attributes.pop("q_func")

        attributes.update({"class": self.__class__})
        
        with open(os.path.join(path, "agent.pkl"), "wb") as f:
            pickle.dump(attributes, f)
        return
    
    def save_rl_parameters(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "rl_params.pkl"), "wb") as f:
            pickle.dump(self.mpc.p_fun(0), f)
    
    def save_q_func(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        self.q_func.save(os.path.join(path, "q_func.keras"))
    
    def load_rl_parameters(self, path: str):
        with open(os.path.join(path, "rl_params.pkl"), "rb") as f:
            rl_params = pickle.load(f)

        p_template = self.mpc.get_p_template(1)
        p_template.master = rl_params["_p", 0]
        self.mpc.set_p_fun(lambda t_now: p_template)
        return self.mpc.p_fun(0)

    def load_q_func(self, path: str):
        self.q_func = load_nn_model(os.path.join(path, "q_func.keras"))
        return self.q_func

    @staticmethod
    def load(path: str, load_differentiator: bool = True):
        with open(os.path.join(path, "agent.pkl"), "rb") as f:
            agent_attributes = pickle.load(f)

        cls = agent_attributes.pop("class")

        mpc = RL_MPC.load(os.path.join(path, "mpc.pkl"))

        rl_settings = agent_attributes.pop("settings")

        agent = cls(mpc, rl_settings.__dict__, init_differentiator = False)
        # agent.differentiator = differentiator
        if load_differentiator:
            with open(os.path.join(path, "differentiator.pkl"), "rb") as f:
                agent.differentiator = pickle.load(f)

        agent.load_q_func(path)
        
        for key, value in agent_attributes.items():
            setattr(agent, key, value)
        return agent

    def save_memories(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        memory = {
            "V_func_transition_memory": self.V_func_transition_memory,
            "Q_func_transition_memory": self.Q_func_transition_memory,
        }
        with open(path, "wb") as f:
            pickle.dump(memory, f)

        self.V_func_transition_memory = []
        self.Q_func_transition_memory = []
        return
        
    def synchronize_memories(self, path: str):
        files = os.listdir(path)

        for file in files:
            if file == "memory.pkl":
                continue

            with open(os.path.join(path, file), "rb") as f:
                memory = pickle.load(f)
            self.V_func_transition_memory = memory["V_func_transition_memory"]
            self.Q_func_transition_memory = memory["Q_func_transition_memory"]
            
            self.remember_episode_for_V_func()
            self.remember_episode_for_Q_func()

            os.remove(os.path.join(path, file))
        
        memory = {
            "episodes_for_state_value_function": self.episodes_for_state_value_function,
            "episodes_for_action_value_function": self.episodes_for_action_value_function
        }
        with open(os.path.join(path, "memory.pkl"), "wb") as f:
            pickle.dump(memory, f)

        return 
    




class RL_MPC_GA_agent(RL_MPC_agent):

    def __init__(self, mpc: RL_MPC, settings_dict: dict = {}, init_differentiator: bool = True):
        super().__init__(mpc, settings_dict)

        if init_differentiator:
            self.differentiator = NLP_differentiator(self.mpc)
            self.flags.differentiator_initialized = True
        else:
            self.differentiator = None
            self.flags.differentiator_initialized = False

        if self.settings.use_momentum:
            if self.settings.use_adam:
                raise ValueError("Momentum and Adam cannot be used together. Please choose one of them.")
            
            self.m = np.zeros((self.mpc.model._p.shape[0], 1))
            self.update_counter = 0

        elif self.settings.use_adam:
            self.m = np.zeros((self.mpc.model._p.shape[0], 1))
            self.v = np.zeros((self.mpc.model._p.shape[0], 1))
            self.update_counter = 0

    def act(self, state: np.ndarray, old_action: np.ndarray = None, training: bool = False):
        action = self.mpc.make_step(state, old_action)

        if not training:
            return action
        
        if not self.flags.differentiator_initialized:
            raise ValueError("The differentiator must be initialized before training.")
        if not self.mpc.solver_stats["success"]:
            print("\nThe solver did not converge. The action is not used for training.")
        
        if self.mpc.solver_stats["success"]:
            jac_action_parameters = self.differentiator.jac_action_parameters(self.mpc)
        else:
            jac_action_parameters = cd.DM.zeros((self.mpc.model._u.shape[0], self.mpc.model._p.shape[0]))

        action_dict = {
            "action": action,
            "jac_action_parameters": jac_action_parameters,
            "success": self.mpc.solver_stats["success"],
        }

        return action_dict
    
    def _prepare_calculations(self):

        ### First do everything for the state value function
        observed_states = []
        observed_previous_actions = []

        observed_taken_action = []
        observed_jac_taken_action_parameters = []

        observed_rewards = []

        observed_next_state = []

        observed_v_values = []

        observed_termination = []
        observed_truncation = []

        for episode in self.episodes_for_state_value_function:
            for idx, (state, previous_action, taken_action, jac_action_parameters, reward, next_state, termination, truncation) in enumerate(reversed(episode)):
                
                if idx == 0:
                    v_value = reward
                else:
                    v_value = reward + self.settings.gamma * v_value

                observed_states.append(state)
                observed_previous_actions.append(previous_action)

                observed_taken_action.append(taken_action)
                observed_jac_taken_action_parameters.append(jac_action_parameters)

                observed_rewards.append(reward)

                observed_next_state.append(next_state)

                observed_v_values.append(v_value)

                observed_termination.append(termination)
                observed_truncation.append(truncation)
            pass

        self.observed_states = np.hstack(observed_states).T
        self.observed_previous_actions = np.hstack(observed_previous_actions).T

        self.observed_taken_actions = np.hstack(observed_taken_action).T
        self.observed_jac_taken_action_parameters = np.stack(observed_jac_taken_action_parameters)

        self.observed_rewards = np.array(observed_rewards).reshape(-1, 1)

        self.observed_next_state = np.hstack(observed_next_state).T

        self.observed_v_values = np.array(observed_v_values).reshape(-1, 1)

        self.observed_termination = np.array(observed_termination).reshape(-1, 1)
        self.observed_truncation = np.array(observed_truncation).reshape(-1, 1)



        ### Now do everything for the action value function
        explored_states = []
        explored_previous_actions = []

        explored_taken_actions = []

        explored_rewards = []

        explored_next_states = []

        explored_termination = []
        explored_truncation = []

        for episode in self.episodes_for_action_value_function:
            for idx, (state, previous_action, taken_action, reward, next_state, termination, truncation) in enumerate(reversed(episode)):
                explored_states.append(state)
                explored_previous_actions.append(previous_action)

                explored_taken_actions.append(taken_action)

                explored_rewards.append(reward)

                explored_next_states.append(next_state)

                explored_termination.append(termination)
                explored_truncation.append(truncation)

        self.explored_states = np.hstack(explored_states).T
        self.explored_previous_actions = np.hstack(explored_previous_actions).T

        self.explored_taken_actions = np.hstack(explored_taken_actions).T

        self.explored_rewards = np.array(explored_rewards).reshape(-1, 1)

        self.explored_next_states = np.hstack(explored_next_states).T

        self.explored_termination = np.array(explored_termination).reshape(-1, 1)
        self.explored_truncation = np.array(explored_truncation).reshape(-1, 1)

        return
    
    def replay(self):
        
        self._prepare_calculations()

        if self.settings.use_scaled_actions:
            self._scale_actions()

        if self.settings.q_func_always_reset:
            # Reset the q_func model
            self.q_func = self._prepare_q_func()

        
        self._learn_q_function(
            self.observed_states,
            self.observed_previous_actions,
            self.observed_taken_actions,
            self.observed_rewards,
            self.observed_v_values,
            self.observed_next_state,
            self.observed_termination,
            self.explored_states,
            self.explored_previous_actions,
            self.explored_taken_actions,
            self.explored_rewards,
            self.explored_next_states,
            self.explored_termination,
            )

        # Get everything for the policy gradient
        a_values, grad_Q_a = self._get_Q_gradient(self.observed_states, self.observed_previous_actions, self.observed_taken_actions)
        
        if self.settings.clip_jac_policy:
            where_at_upper_bound = np.where(np.isclose(self.observed_taken_actions - self.mpc._u_ub.master.T.full(), 0, atol = 1e-6))
            self.observed_jac_taken_action_parameters[where_at_upper_bound] = np.clip(self.observed_jac_taken_action_parameters[where_at_upper_bound], a_max = 0, a_min = None)

            where_at_lower_bound = np.where(np.isclose(self.observed_taken_actions - self.mpc._u_lb.master.T.full(), 0, atol = 1e-6))
            self.observed_jac_taken_action_parameters[where_at_lower_bound] = np.clip(self.observed_jac_taken_action_parameters[where_at_lower_bound], a_min = 0, a_max = None)
        
        policy_gradient = self.compute_policy_gradient(self.observed_jac_taken_action_parameters, grad_Q_a)
        print(f"Policy gradient: {policy_gradient}")
        
        update = self._compute_update(policy_gradient)

        self._update_parameters(update)

        self.episodes_for_action_value_function = []
        self.episodes_for_state_value_function = []

        return policy_gradient
    
    def _get_Q_gradient(self, observed_states: np.ndarray, observed_previous_actions: np.ndarray, observed_taken_actions: np.ndarray):

        scaled_states = self.observed_states_scaler.transform(observed_states)
        scaled_previous_actions = self.observed_previous_actions_scaler.transform(observed_previous_actions)
        exploration = np.zeros(observed_previous_actions.shape)

        scaled_states = constant(scaled_states, dtype = np.float64)
        scaled_previous_actions = constant(scaled_previous_actions, dtype = np.float64)
        observed_taken_actions = constant(observed_taken_actions, dtype = np.float64)
        exploration = constant(exploration, dtype = np.float64)

        action_min = constant(self.observed_previous_actions_scaler.min_, dtype = np.float64)
        action_scale = constant(self.observed_previous_actions_scaler.scale_, dtype = np.float64)
        v_func_min = constant(self.observed_v_values_scaler.min_, dtype = np.float64)
        v_func_scale = constant(self.observed_v_values_scaler.scale_, dtype = np.float64)

        # Compute the gradient of the action-value function
        with GradientTape(persistent=False) as tape:
            tape.watch(observed_taken_actions)

            scaled_observed_taken_actions = action_scale * observed_taken_actions + action_min

            scaled_v_a_values = self.q_func([scaled_states, scaled_previous_actions, scaled_observed_taken_actions], training = False)

            scaled_v_values = scaled_v_a_values[:, 0:1]  # Get the state value function output
            scaled_a_values = scaled_v_a_values[:, 1:2]  # Get the action value function output

            v_values = (scaled_v_values - v_func_min) / v_func_scale
            a_values = scaled_a_values / v_func_scale 


        grad_Q_a = tape.gradient(a_values, observed_taken_actions) # This is the same as the Q-function gradient because V is not a function of a
        
        v_values = v_values.numpy()
        grad_Q_a = grad_Q_a.numpy()

        if self.settings.clip_q_gradients:
            where_at_upper_bound = np.where(np.isclose(observed_taken_actions - self.mpc._u_ub.master.T.full(), 0, atol = 1e-6))
            grad_Q_a[where_at_upper_bound] = np.clip(grad_Q_a[where_at_upper_bound], a_max = 0, a_min = None)

            where_at_lower_bound = np.where(np.isclose(observed_taken_actions - self.mpc._u_lb.master.T.full(), 0, atol = 1e-6))
            grad_Q_a[where_at_lower_bound] = np.clip(grad_Q_a[where_at_lower_bound], a_min = 0, a_max = None)

        return v_values, grad_Q_a

    def _compute_update(self, policy_gradient: np.ndarray):
        """
        This function is used to compute the update for the parameters.
        """
        if self.settings.use_momentum:
            self.update_counter += 1

            # Update biased first moment estimate
            self.m = self.settings.adam_beta_1 * self.m + (1 - self.settings.adam_beta_1) * policy_gradient

            # Compute bias-corrected first moment estimate
            m_hat = self.m / (1 - self.settings.adam_beta_1 ** self.update_counter)

            update = m_hat * self.settings.actor_learning_rate

        elif self.settings.use_adam:
            self.update_counter += 1

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

        return update




class RL_MPC_GN_agent(RL_MPC_GA_agent):

    def __init__(self, mpc: RL_MPC, settings_dict: dict = {}, init_differentiator: bool = True):
        super().__init__(mpc, settings_dict, init_differentiator)

        # Initialize the GN estimate
        if self.settings.use_momentum:
            self.D_init = -np.eye(self.mpc.model._p.shape[0]) * self.settings.omegainv
            self.D = self.D_init.copy()

        elif self.settings.use_adam:
            raise ValueError("You try to use Adam for second order optimization. Adam already scaled the first order update so the gradient is theoretically already corrected.")
            

    def replay(self):
        self._prepare_calculations()

        if self.settings.use_scaled_actions:
            self._scale_actions()

        if self.settings.q_func_always_reset:
            # Reset the q_func model
            self.q_func = self._prepare_q_func()

        
        self._learn_q_function(
            self.observed_states,
            self.observed_previous_actions,
            self.observed_taken_actions,
            self.observed_rewards,
            self.observed_v_values,
            self.observed_next_state,
            self.observed_termination,
            self.explored_states,
            self.explored_previous_actions,
            self.explored_taken_actions,
            self.explored_rewards,
            self.explored_next_states,
            self.explored_termination,
            )


        # Get everything for the policy gradient
        a_values, grad_Q_a, hess_Q_a = self._get_Q_hessian(self.observed_states, self.observed_previous_actions, self.observed_taken_actions)
        
        if self.settings.clip_jac_policy:
            where_at_upper_bound = np.where(np.isclose(self.observed_taken_actions - self.mpc._u_ub.master.T.full(), 0, atol = 1e-6))
            self.observed_jac_taken_action_parameters[where_at_upper_bound] = np.clip(self.observed_jac_taken_action_parameters[where_at_upper_bound], a_max = 0, a_min = None)

            where_at_lower_bound = np.where(np.isclose(self.observed_taken_actions - self.mpc._u_lb.master.T.full(), 0, atol = 1e-6))
            self.observed_jac_taken_action_parameters[where_at_lower_bound] = np.clip(self.observed_jac_taken_action_parameters[where_at_lower_bound], a_min = 0, a_max = None)
        
        policy_gradient = self.compute_policy_gradient(self.observed_jac_taken_action_parameters, grad_Q_a)
        policy_hessian = self.compute_Gauss_Newton_matrix(self.observed_jac_taken_action_parameters, hess_Q_a)

        policy_hessian = self._regularize_policy_hessian(policy_hessian, self.observed_jac_taken_action_parameters)
        print(f"Policy gradient: {policy_gradient}")
        print(f"Policy hessian: {policy_hessian}")

        update = self._compute_update(policy_gradient, policy_hessian)

        self._update_parameters(update)

        self.episodes_for_action_value_function = []
        self.episodes_for_state_value_function = []

        return policy_gradient, policy_hessian

    def _get_Q_hessian(self, observed_states: np.ndarray, observed_previous_actions: np.ndarray, observed_taken_actions: np.ndarray):

        scaled_states = self.observed_states_scaler.transform(observed_states)
        scaled_previous_actions = self.observed_previous_actions_scaler.transform(observed_previous_actions)
        observed_taken_actions_tf = constant(observed_taken_actions, dtype = np.float64)
        exploration = np.zeros(observed_previous_actions.shape)

        scaled_states = constant(scaled_states, dtype = np.float64)
        scaled_previous_actions = constant(scaled_previous_actions, dtype = np.float64)
        exploration = constant(exploration, dtype = np.float64)

        action_min = constant(self.observed_previous_actions_scaler.min_, dtype = np.float64)
        action_scale = constant(self.observed_previous_actions_scaler.scale_, dtype = np.float64)
        v_func_scale = constant(self.observed_v_values_scaler.scale_, dtype = np.float64)

        # Compute the gradient of the advantage function
        with GradientTape(persistent=False) as tape_hessian:
            tape_hessian.watch(observed_taken_actions_tf)
            with GradientTape(persistent=False) as tape_gradient:
                tape_gradient.watch(observed_taken_actions_tf)

                scaled_observed_taken_actions_tf = action_scale * observed_taken_actions_tf + action_min 

                scaled_v_values = self.q_func([scaled_states, scaled_previous_actions, scaled_observed_taken_actions_tf], training = False)

                advantages = scaled_v_values[:, 1:2] / v_func_scale

            grad_Q_a = tape_gradient.batch_jacobian(advantages, observed_taken_actions_tf)
    
        hess_Q_a = tape_hessian.batch_jacobian(grad_Q_a, observed_taken_actions_tf)

        advantages = advantages.numpy()
        grad_Q_a = grad_Q_a.numpy()[:, 0, :]
        hess_Q_a = hess_Q_a.numpy()[:, 0, :, :]

        if self.settings.clip_q_gradients:
            where_at_upper_bound = np.where(np.isclose(observed_taken_actions - self.mpc._u_ub.master.T.full(), 0, atol = 1e-6))
            grad_Q_a[where_at_upper_bound] = np.clip(grad_Q_a[where_at_upper_bound], a_max = 0, a_min = None)

            where_at_lower_bound = np.where(np.isclose(observed_taken_actions - self.mpc._u_lb.master.T.full(), 0, atol = 1e-6))
            grad_Q_a[where_at_lower_bound] = np.clip(grad_Q_a[where_at_lower_bound], a_min = 0, a_max = None)

        return advantages, grad_Q_a, hess_Q_a

    def _compute_update(self, policy_gradient: np.ndarray, policy_hessian: np.ndarray):

        if self.settings.use_momentum:
            self.update_counter += 1
            
            # Correct the policy gradient
            self.m = self.settings.momentum_beta * self.m + (1 - self.settings.momentum_beta) * policy_gradient
            m_hat = self.m / (1 - self.settings.momentum_beta ** self.update_counter)

            corrected_policy_gradient = m_hat

            # Correct the policy hessian
            self.D = self.settings.momentum_eta * self.D + (1 - self.settings.momentum_eta) * policy_hessian
                     
            # Bias corrected policy Hessian
            D_hat = self.D / (1 - self.settings.momentum_eta ** (self.update_counter + 1))

            update = np.linalg.solve(D_hat, - corrected_policy_gradient)
            update *= self.settings.actor_learning_rate

        elif self.settings.use_adam:
            raise ValueError("You cannot use Adam with the Gauss-Newton approach.")
        
        else:
            update = np.linalg.solve(policy_hessian, - policy_gradient)
            update *= self.settings.actor_learning_rate
        return update
    


class RL_MPC_AN_agent(RL_MPC_GN_agent):

    def __init__(self, mpc: RL_MPC, settings_dict: dict = {}, init_differentiator: bool = True):
        super().__init__(mpc, settings_dict, False)

        if init_differentiator:
            self.differentiator = NLP_differentiator(self.mpc, second_order = True)
            self.flags.differentiator_initialized = True

    def act(self, state: np.ndarray, old_action: np.ndarray,  training: bool = False):
        action = self.mpc.make_step(state, old_action)

        if not training:
            return action
        
        if not self.flags.differentiator_initialized:
            raise ValueError("The differentiator must be initialized before training.")
        if not self.mpc.solver_stats["success"]:
            print("\nThe solver did not converge. The action is not used for training.")
        
        if self.mpc.solver_stats["success"]:
            jac_action_parameters, jac_jac_action_parameters = self.differentiator.jac_jac_action_parameters_parameters(self.mpc)
        else:
            jac_action_parameters = np.zeros((self.mpc.model._u.shape[0], self.mpc.model._p.shape[0]))
            jac_jac_action_parameters = np.zeros((self.mpc.model._u.shape[0], self.mpc.model._p.shape[0], self.mpc.model._p.shape[0]))

        action_dict = {
            "action": action,
            "jac_action_parameters": jac_action_parameters,
            "jac_jac_action_parameters": jac_jac_action_parameters,
            "success": self.mpc.solver_stats["success"],
        }

        return action_dict
    
    def remember_transition_for_V_func(
            self,
            state: np.ndarray,
            previous_action: np.ndarray,
            taken_action: np.ndarray,
            jac_action_parameters:np.ndarray,
            jac_jac_action_parameters: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            termination: bool,
            truncation: bool
            ):
        self.V_func_transition_memory.append(
            (
                state,
                previous_action,
                taken_action,
                jac_action_parameters,
                jac_jac_action_parameters,
                reward,
                next_state,
                termination,
                truncation,
            )
        )

    def _prepare_calculations(self):

        ### First do everything for the state value function
        observed_states = []
        observed_previous_actions = []

        observed_taken_action = []
        observed_jac_taken_action_parameters = []
        observed_jac_jac_taken_action_parameters = []

        observed_rewards = []

        observed_next_state = []

        observed_v_values = []

        observed_termination = []
        observed_truncation = []

        for episode in self.episodes_for_state_value_function:
            for idx, (state, previous_action, taken_action, jac_action_parameters, jac_jac_action_parameters, reward, next_state, termination, truncation) in enumerate(reversed(episode)):
                
                if idx == 0:
                    v_value = reward
                else:
                    v_value = reward + self.settings.gamma * v_value

                observed_states.append(state)
                observed_previous_actions.append(previous_action)

                observed_taken_action.append(taken_action)
                observed_jac_taken_action_parameters.append(jac_action_parameters)
                observed_jac_jac_taken_action_parameters.append(jac_jac_action_parameters)

                observed_rewards.append(reward)

                observed_next_state.append(next_state)

                observed_v_values.append(v_value)

                observed_termination.append(termination)
                observed_truncation.append(truncation)
            pass

        self.observed_states = np.hstack(observed_states).T
        self.observed_previous_actions = np.hstack(observed_previous_actions).T

        self.observed_taken_actions = np.hstack(observed_taken_action).T
        self.observed_jac_taken_action_parameters = np.stack(observed_jac_taken_action_parameters)
        self.observed_jac_jac_taken_action_parameters = np.stack(observed_jac_jac_taken_action_parameters)

        self.observed_rewards = np.array(observed_rewards).reshape(-1, 1)

        self.observed_next_state = np.hstack(observed_next_state).T

        self.observed_v_values = np.array(observed_v_values).reshape(-1, 1)

        self.observed_termination = np.array(observed_termination).reshape(-1, 1)
        self.observed_truncation = np.array(observed_truncation).reshape(-1, 1)



        ### Now do everything for the action value function
        explored_states = []
        explored_previous_actions = []

        explored_taken_actions = []

        explored_rewards = []

        explored_next_states = []

        explored_termination = []
        explored_truncation = []

        for episode in self.episodes_for_action_value_function:
            for idx, (state, previous_action, taken_action, reward, next_state, termination, truncation) in enumerate(reversed(episode)):
                explored_states.append(state)
                explored_previous_actions.append(previous_action)

                explored_taken_actions.append(taken_action)

                explored_rewards.append(reward)

                explored_next_states.append(next_state)

                explored_termination.append(termination)
                explored_truncation.append(truncation)

        self.explored_states = np.hstack(explored_states).T
        self.explored_previous_actions = np.hstack(explored_previous_actions).T

        self.explored_taken_actions = np.hstack(explored_taken_actions).T

        self.explored_rewards = np.array(explored_rewards).reshape(-1, 1)

        self.explored_next_states = np.hstack(explored_next_states).T

        self.explored_termination = np.array(explored_termination).reshape(-1, 1)
        self.explored_truncation = np.array(explored_truncation).reshape(-1, 1)

        return
    
    def replay(self):
        self._prepare_calculations()

        if self.settings.use_scaled_actions:
            self._scale_actions()

        if self.settings.q_func_always_reset:
            # Reset the q_func model
            self.q_func = self._prepare_q_func()

        
        self._learn_q_function(
            self.observed_states,
            self.observed_previous_actions,
            self.observed_taken_actions,
            self.observed_rewards,
            self.observed_v_values,
            self.observed_next_state,
            self.observed_termination,
            self.explored_states,
            self.explored_previous_actions,
            self.explored_taken_actions,
            self.explored_rewards,
            self.explored_next_states,
            self.explored_termination,
            )


        # Get everything for the policy gradient
        a_values, grad_Q_a, hess_Q_a = self._get_Q_hessian(self.observed_states, self.observed_previous_actions, self.observed_taken_actions)

        if self.settings.clip_jac_policy:
            where_at_upper_bound = np.where(np.isclose(self.observed_taken_actions - self.mpc._u_ub.master.T.full(), 0, atol = 1e-6))
            self.observed_jac_taken_action_parameters[where_at_upper_bound] = np.clip(self.observed_jac_taken_action_parameters[where_at_upper_bound], a_max = 0, a_min = None)

            where_at_lower_bound = np.where(np.isclose(self.observed_taken_actions - self.mpc._u_lb.master.T.full(), 0, atol = 1e-6))
            self.observed_jac_taken_action_parameters[where_at_lower_bound] = np.clip(self.observed_jac_taken_action_parameters[where_at_lower_bound], a_min = 0, a_max = None)


        policy_gradient = self.compute_policy_gradient(self.observed_jac_taken_action_parameters, grad_Q_a)
        policy_hessian = self.compute_approximate_Newton_matrix(self.observed_jac_taken_action_parameters, self.observed_jac_jac_taken_action_parameters, grad_Q_a, hess_Q_a)

        policy_hessian = self._regularize_policy_hessian(policy_hessian, self.observed_jac_taken_action_parameters)
        print(f"Policy gradient: {policy_gradient}")
        print(f"Policy hessian: {policy_hessian}")

        update = self._compute_update(policy_gradient, policy_hessian)

        self._update_parameters(update)

        self.episodes_for_action_value_function = []
        self.episodes_for_state_value_function = []

        return policy_gradient, policy_hessian
