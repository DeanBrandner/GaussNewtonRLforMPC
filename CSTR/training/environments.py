import gymnasium as gym
import casadi as cd
import numpy as np
from dataclasses import dataclass
from collections import deque

# Settings for the CSTR environment
@dataclass
class CSTRSettings:
    seed: int = None
    terminate_on_cv: bool = True
    max_steps: int = 50
    max_steps_of_violation: int = 5

# Bounds for process variables and controls
@dataclass
class ProcessBounds:
    x_lb: np.ndarray = None
    x_ub: np.ndarray = None
    u_lb: np.ndarray = None
    u_ub: np.ndarray = None

# Class to store history of environment interactions
class History:
    def __init__(self) -> None:
        self.idx = []
        self.x = []
        self.u = []
        self.time = []
        self.r = []
        self.stage_cost = []
        self.penalty = []
        self.x_next = []

    # Store a new entry in history
    def remember(self, x, u, time, r, stage_cost, penalty, x_next) -> None:
        self.idx.append(len(self.idx))
        self.x.append(x)
        self.u.append(u)
        self.time.append(time)
        self.r.append(r)
        self.stage_cost.append(stage_cost)
        self.penalty.append(penalty)
        self.x_next.append(x_next)

    # Convert lists to numpy arrays for efficient processing
    def compactify(self) -> None:
        self.idx = np.array(self.idx)
        self.x = np.array(self.x)
        self.u = np.array(self.u)
        self.time = np.array(self.time)
        self.r = np.array(self.r)
        self.stage_cost = np.array(self.stage_cost)
        self.penalty = np.array(self.penalty)
        self.x_next = np.array(self.x_next)

    # String representation
    def __repr__(self):
        current_data_string = f"History contains {len(self.idx)} entries.\n"
        if len(self.idx) > 0:
            current_data_string += "Last entry:\n"
            current_data_string += f"Index: {self.idx[-1]}\n"
            current_data_string += f"x: {self.x[-1]}\n"
            current_data_string += f"u: {self.u[-1]}\n"
            current_data_string += f"time: {self.time[-1]}\n"
            current_data_string += f"r: {self.r[-1]}\n"
            current_data_string += f"stage_cost: {self.stage_cost[-1]}\n"
            current_data_string += f"penalty: {self.penalty[-1]}\n"
            current_data_string += f"x_next: {self.x_next[-1]}\n"
        else:
            current_data_string += "No entries recorded yet."
        return current_data_string

# Main CSTR environment class
class CSTR(gym.Env):

    def __init__(self, seed:int = 1234, terminate_on_cv: bool = True, max_steps: int = 100, max_steps_of_violation:int=3) -> None:
        super().__init__()

        # Set environment settings
        self.settings = CSTRSettings(
            seed=seed,
            terminate_on_cv=terminate_on_cv,
            max_steps=max_steps,
            max_steps_of_violation=max_steps_of_violation
            )

        # Random number generators for initial state and uncertain parameters
        self.rng = np.random.default_rng(seed)
        self.rng_uncertain_params = np.random.default_rng(seed + 1)

        # Setup system matrices and integrator
        self._setup_system()

        # Define action and observation spaces for RL
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=float)
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(6,), dtype=float)

        # Initialize process, initial, and terminal bounds
        self._init_process_bounds()
        self._init_init_bounds()
        self._init_term_bounds()

        # Initialize history and termination/truncation buffers
        self.data = History()
        self.last_terminations = deque(maxlen = 5)
        self.last_truncations = deque(maxlen = self.settings.max_steps_of_violation)

        # Initialize state, parameters, and time
        self.x_num = self.sample_initial_state()
        self.uncertain_params = self.sample_uncertain_params()
        self.current_step = 0
        self.time = 0.0
        self.dt = 0.005  # Integration time step

    # Set process bounds for states and controls
    def _init_process_bounds(self) -> None:
        self.bounds = ProcessBounds(
            x_lb=np.array([0.1, 0.1, 80.0, 80.0,   5.0, -8500.0]).reshape(-1, 1),
            x_ub=np.array([2.0, 2.0,  140,  140, 40.0, 0.0]).reshape(-1, 1),
            u_lb=np.array([5.0, -8500]).reshape(-1, 1),
            u_ub=np.array([40.0, 0.0]).reshape(-1, 1),
        )

    # Set initial bounds for sampling initial state
    def _init_init_bounds(self) -> None:
        self.init_bounds = ProcessBounds(
            x_lb=np.array([0.1, 0.1, 80.0, 80.0,  5.0, -8500.0]).reshape(-1, 1),
            x_ub=np.array([2.0, 2.0,  140,  140, 40.0, 0.0]).reshape(-1, 1),
            u_lb=np.array([5.0, -8500]).reshape(-1, 1),
            u_ub=np.array([40.0, 0.0]).reshape(-1, 1),
        )

    # Set terminal bounds for termination condition
    def _init_term_bounds(self) -> None:
        self.term_bounds = ProcessBounds(
            x_lb=np.array([0.1, 0.1, 126 - 1, 120.0 - 1, 5, -8500.0]).reshape(-1, 1),
            x_ub=np.array([2.0, 2.0, 126 + 1, 120.0 + 1, 40,     0.0]).reshape(-1, 1),
            u_lb=np.array([5, -8500.0]).reshape(-1, 1),
            u_ub=np.array([40,     0.0]).reshape(-1, 1),
        )

    # Setup CasADi system for CSTR dynamics and integrator
    def _setup_system(self) -> None:
        # Define state variables
        C_a = cd.SX.sym("C_a")
        C_b = cd.SX.sym("C_b")
        T_R = cd.SX.sym("T_R")
        T_K = cd.SX.sym("T_K")
        F_prev = cd.SX.sym("F_prev")
        Q_dot_prev = cd.SX.sym("Q_dot_prev")

        states = cd.vertcat(C_a, C_b, T_R, T_K)
        self.state_func = cd.Function("state_func", [C_a, C_b, T_R, T_K], [states], ["C_a", "C_b", "T_R", "T_K"], ["states"])
        self.state_inv_func = cd.Function("state_inv_func", [states], [C_a, C_b, T_R, T_K], ["states"], ["C_a", "C_b", "T_R", "T_K"])
        
        alg_states = cd.vertcat(F_prev, Q_dot_prev)
        self.alg_state_func = cd.Function("alg_state_func", [F_prev, Q_dot_prev], [alg_states], ["F_prev", "Q_dot_prev"], ["alg_states"])
        self.alg_state_inv_func = cd.Function("alg_state_inv_func", [alg_states], [F_prev, Q_dot_prev], ["alg_states"], ["F_prev", "Q_dot_prev"])

        all_states = cd.vertcat(states, alg_states)
        self.all_state_func = cd.Function("all_state_func", [C_a, C_b, T_R, T_K, F_prev, Q_dot_prev], [all_states], ["C_a", "C_b", "T_R", "T_K", "F_prev", "Q_dot_prev"], ["all_states"])
        self.all_state_inv_func = cd.Function("all_state_inv_func", [all_states], [C_a, C_b, T_R, T_K, F_prev, Q_dot_prev], ["all_states"], ["C_a", "C_b", "T_R", "T_K", "F_prev", "Q_dot_prev"])

        # Define control variables
        F = cd.SX.sym("F")
        Q_dot = cd.SX.sym("Q_dot")
        controls = cd.vertcat(F, Q_dot)
        self.control_func = cd.Function("control_func", [F, Q_dot], [controls], ["F", "Q_dot"], ["controls"])
        self.control_inv_func = cd.Function("control_inv_func", [controls], [F, Q_dot], ["controls"], ["F", "Q_dot"])

        # Define constants for CSTR model
        K0_ab = 1.287e12 # K0 [h^-1]
        K0_bc = 1.287e12 # K0 [h^-1]
        K0_ad = 9.043e9 # K0 [l/mol.h]
        R_gas = 8.3144621e-3 # Universal gas constant
        E_A_ab = 9758.3*1.00 #* R_gas# [kj/mol]
        E_A_bc = 9758.3*1.00 #* R_gas# [kj/mol]
        E_A_ad = 8560.0*1.0 #* R_gas# [kj/mol]
        H_R_ab = 4.2 # [kj/mol A]
        H_R_bc = -11.0 # [kj/mol B] Exothermic
        H_R_ad = -41.85 # [kj/mol A] Exothermic
        Rou = 0.9342 # Density [kg/l]
        Cp = 3.01 # Specific Heat capacity [kj/Kg.K]
        Cp_k = 2.0 # Coolant heat capacity [kj/kg.k]
        A_R = 0.215 # Area of reactor wall [m^2]
        V_R = 10.01 #0.01 # Volume of reactor [l]
        m_k = 5.0 # Coolant mass[kg]
        T_in = 130.0 # Temp of inflow [Celsius]
        K_w = 4032.0 # [kj/h.m^2.K]
        C_A0 = (5.7+4.5)/2.0*1.0

        # Uncertain parameters (mu_alpha and mu_beta in paper)
        alpha = cd.SX.sym("alpha")
        beta = cd.SX.sym("beta")
        uncertain_params = cd.vertcat(alpha, beta)

        # Reaction rate expressions
        K_1 = beta * K0_ab * cd.exp((-E_A_ab)/((T_R+273.15)))
        K_2 =  K0_bc * cd.exp((-E_A_bc)/((T_R+273.15)))
        K_3 = K0_ad * cd.exp((-alpha*E_A_ad)/((T_R+273.15)))

        T_dif = T_R - T_K

        # Differential equations for CSTR
        dCa_dt = F * (C_A0 - C_a) - K_1 * C_a - K_3 * (C_a ** 2)
        dCb_dt = - F * C_b + K_1 * C_a - K_2 * C_b
        dTR_dt = ((K_1*C_a*H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2)*H_R_ad)/(-Rou*Cp)) + F*(T_in-T_R) +(((K_w*A_R)*(-T_dif))/(Rou*Cp*V_R))
        dTK_dt = (Q_dot + K_w*A_R*(T_dif))/(m_k*Cp_k)

        dstates_dt = cd.vertcat(dCa_dt, dCb_dt, dTR_dt, dTK_dt)

        # Algebraic equations for previous controls
        F_prev_eq = F - F_prev
        Q_dot_prev_eq = Q_dot - Q_dot_prev
        alg_eq = cd.vertcat(F_prev_eq, Q_dot_prev_eq)

        # Parameters for integration
        p_integration = cd.vertcat(controls, uncertain_params)

        # DAE dictionary for CasADi integrator
        dae_dict = {
            "x": states,
            "z": alg_states,
            "p": p_integration,
            "ode": dstates_dt,
            "alg": alg_eq,
        }

        # Create CasADi integrator for system simulation
        self.integrator = cd.integrator("CSTR", "idas", dae_dict, 0.0, 0.005)
        self.integrator = self.integrator.factory("CSTR", ["x0", "z0", "p"], ["xf", "zf"])
        return

    # Step function for RL environment
    def step(self, action: np.ndarray, scaled_action: bool = True, scale_observation: bool = True) -> tuple[np.ndarray, float, bool, bool, dict]:
        info = {}

        x = self.x_num.copy()
        action = action.reshape(-1, 1)

        # Scale action from [0,1] to physical bounds
        if scaled_action:
            action = self.bounds.u_lb + (self.bounds.u_ub - self.bounds.u_lb) * action

        # Integrate system dynamics using CasADi
        C_a, C_b, T_R, T_K, F_prev, Q_dot_prev = self.all_state_inv_func(x)
        states, alg_states = self.state_func(C_a, C_b, T_R, T_K), self.alg_state_func(F_prev, Q_dot_prev)
        integration_params = cd.vertcat(action, self.uncertain_params)
        x_next, z_next = self.integrator(states, alg_states, integration_params)
        x_next = x_next.full()
        z_next = z_next.full()

        # Check for termination/truncation conditions
        local_termination, local_truncation, time_exceeded = self._local_termination_truncation_check(x, action)
        self.last_terminations.append(local_termination)
        self.last_truncations.append(local_truncation)
        terminated, truncated = self._termination_truncation_check()

        info["time_exceeded"] = time_exceeded
        info["terminated"] = terminated
        info["truncated"] = truncated

        # Compute reward, stage cost, and penalty
        reward, stage_cost, penalty = self._get_reward(x, action)

        # Update state
        self.x_num = observation = np.vstack((x_next, z_next)).copy()

        # Store transition in history
        self.data.remember(
            x = x,
            u = action,
            time = self.time,
            r = reward,
            stage_cost = stage_cost,
            penalty = penalty,
            x_next = observation,
            )

        # Scale observation to [0,1] if required
        if scale_observation:
            observation = (observation - self.bounds.x_lb) / (self.bounds.x_ub - self.bounds.x_lb)

        self.current_step += 1
        self.time += self.dt

        return observation.T, reward, terminated, truncated, info

    # Compute reward, stage cost, and penalty for RL
    def _get_reward(self, state: np.ndarray, action: np.ndarray) -> tuple[float, float, float]:
        reward = 0.0
        C_a, C_b, T_R, T_K, F_prev, Q_dot_prev = self.all_state_inv_func(state)
        F, Q_dot = self.control_inv_func(action)
        stage_cost = 1e-2 * (T_R - 126) ** 2 + 1e-2 *(T_K - 120) ** 2

        R = 1e-2 *cd.diag([0.1, 0.1])
        delta_action = state[-2:, :] - action
        delta_action = delta_action / (self.bounds.u_ub - self.bounds.u_lb)
        stage_cost += delta_action.T @ R @ delta_action
        
        reward -= stage_cost[0, 0]

        # Constraint violations for states
        weights = np.ones(state.shape) * 100
        cv_xlb = np.max([self.bounds.x_lb - state, np.zeros(state.shape)], axis = 0)
        cv_xub = np.max([state - self.bounds.x_ub, np.zeros(state.shape)], axis = 0)
        penalty = weights.T @ (cv_xlb + cv_xub)
        penalty = penalty[0, 0]

        # Constraint violations for controls
        weights_u = np.ones(action.shape) * 1000
        cv_ulb = np.max([(self.bounds.u_lb - action)/(self.bounds.u_ub - self.bounds.u_lb), np.zeros(action.shape)], axis = 0)
        cv_uub = np.max([(action - self.bounds.u_ub)/(self.bounds.u_ub - self.bounds.u_lb), np.zeros(action.shape)], axis = 0)
        penalty += weights_u.T @ (cv_ulb + cv_uub)
        penalty = penalty[0, 0]

        reward -= penalty

        return reward, stage_cost, penalty
    
    # Check for local termination/truncation conditions
    def _local_termination_truncation_check(self, state: np.ndarray, action: np.ndarray) -> tuple[bool, bool]:
        terminated = False
        truncated = False
        time_exceeded = False

        # Check if max steps exceeded
        if self.current_step >= self.settings.max_steps:
            time_exceeded = True
            truncated = False
            terminated = True
            return terminated, truncated, time_exceeded

        # Check for constraint violations
        viollb = np.max([self.bounds.x_lb - state, np.zeros(state.shape)], axis=0)
        violub = np.max([state - self.bounds.x_ub, np.zeros(state.shape)], axis=0)
        cv_limit = 1e0  # Constraint violation limit
        if np.any(viollb > cv_limit) or np.any(violub > cv_limit):
            if self.settings.terminate_on_cv:
                truncated = True
            return terminated, truncated, time_exceeded

        # Check if state is within terminal bounds
        terminallb = np.max([self.term_bounds.x_lb - state, np.zeros(state.shape)], axis=0)
        terminalub = np.max([state - self.term_bounds.x_ub, np.zeros(state.shape)], axis=0)
        if np.all(terminallb <= 0) and np.all(terminalub <= 0):
            terminated = True
            return terminated, truncated, time_exceeded
        
        return terminated, truncated, time_exceeded
    
    # Check for overall termination/truncation based on history
    def _termination_truncation_check(self) -> tuple[bool, bool]:
        terminated = False
        truncated = False

        if sum(self.last_terminations) == self.last_terminations.maxlen and sum(self.last_truncations) == 0:
            terminated = True
            truncated = False
            return terminated, truncated
        
        if self.settings.terminate_on_cv and sum(self.last_truncations) == self.last_truncations.maxlen:
            truncated = True
            terminated = False
            return terminated, truncated

        return terminated, truncated

    # Set observation (state) manually, with optional scaling
    def set_observation(self, observation: np.ndarray, scale_observation: bool = True) -> np.ndarray:
        if scale_observation:
            observation = self.bounds.x_lb + (self.bounds.x_ub - self.bounds.x_lb) * observation

        if observation.ndim == 1:
            if observation.shape[0] == 2:
                observation = observation.reshape(-1, 1)
            else:
                raise ValueError("Observation must have shape (2,) or (2, 1).")
        elif observation.shape[1] == 2 and observation.shape[0] == 1:
            observation = observation.T

        self.x_num = observation.T.copy()
        return observation

    # Sample initial state from initial bounds
    def sample_initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        x0 = self.rng.uniform(self.init_bounds.x_lb, self.init_bounds.x_ub)
        return x0
    
    # Sample uncertain parameters for the system
    def sample_uncertain_params(self) -> np.ndarray:
        alpha_max = 1.05
        alpha_min = 0.95
        alpha = self.rng_uncertain_params.uniform(low=alpha_min, high=alpha_max)

        beta_max = 1.05
        beta_min = 0.95
        beta = self.rng_uncertain_params.uniform(low=beta_min, high=beta_max)

        uncertain_params = np.vstack((alpha, beta)).reshape(-1, 1)
        return uncertain_params

    # Reset environment to initial state
    def reset(self, seed: int = None, scale_observation: bool = True) -> tuple[np.ndarray, dict]:
        info = {}
        # Update RNGs if new seed is provided
        if seed is not None:
            self.settings.seed = seed
            self.rng = np.random.default_rng(seed)
            self.rng_uncertain_params = np.random.default_rng(seed + 1)

        # Sample initial state and parameters
        self.x_num = self.sample_initial_state()
        self.uncertain_params = self.sample_uncertain_params()
        self.current_step = 0
        self.time = 0.0

        observation = self.x_num.copy()
        if scale_observation:
            observation = (observation - self.bounds.x_lb) / (self.bounds.x_ub - self.bounds.x_lb)

        # Reset history and termination/truncation buffers
        self.data = History()
        self.last_terminations = deque(maxlen = 5)
        self.last_truncations = deque(maxlen = self.settings.max_steps_of_violation)

        return observation.T, info
    
# Example usage and test
if __name__ == "__main__":
    env = CSTR(seed=1234, terminate_on_cv=True, max_steps=100, max_steps_of_violation=5)
    obs, info = env.reset()

    observation, reward, terminated, truncated, info = env.step(np.array([0.5, 0.5]), scaled_action=True, scale_observation=True)