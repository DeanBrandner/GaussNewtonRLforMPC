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
    gamma: float = 0.99

# Bounds for process variables and controls
@dataclass
class ProcessBounds:
    x_lb: np.ndarray = None
    x_ub: np.ndarray = None
    u_lb: np.ndarray = None
    u_ub: np.ndarray = None

# Class to store history of environment interactions
class History:
    def __init__(self, x_init: np.ndarray = None, y_init: np.ndarray = None, time: float = None) -> None:
        self.idx = []
        self.x = [x_init]
        self.y = [y_init]
        self.u = []
        self.time = [time]
        self.r = []
        self.stage_cost = []
        self.penalty = []
        self.x_next = []
        self.y_next = []

        self.flags = {
            "compactified": False
        }

    # Store a new entry in history
    def remember(self, x, y, u, time, r, stage_cost, penalty, x_next, y_next) -> None:
        self.idx.append(len(self.idx))
        self.x.append(x)
        self.y.append(y)
        self.u.append(u)
        self.time.append(time)
        self.r.append(r)
        self.stage_cost.append(stage_cost)
        self.penalty.append(penalty)
        self.x_next.append(x_next)
        self.y_next.append(y_next)

    # Convert lists to numpy arrays for efficient processing
    def compactify(self) -> None:
        self.idx = np.array(self.idx)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.u = np.array(self.u)
        self.time = np.array(self.time)
        self.r = np.array(self.r)
        self.stage_cost = np.array(self.stage_cost)
        self.penalty = np.array(self.penalty)
        self.x_next = np.array(self.x_next)
        self.y_next = np.array(self.y_next)

        self.flags["compactified"] = True

    # String representation
    def __repr__(self):
        current_data_string = f"History contains {len(self.idx)} entries.\n"
        if len(self.idx) > 0:
            current_data_string += "Last entry:\n"
            current_data_string += f"Index: {self.idx[-1]}\n"
            current_data_string += f"x: {self.x[-1]}\n"
            current_data_string += f"y: {self.y[-1]}\n"
            current_data_string += f"u: {self.u[-1]}\n"
            current_data_string += f"time: {self.time[-1]}\n"
            current_data_string += f"r: {self.r[-1]}\n"
            current_data_string += f"stage_cost: {self.stage_cost[-1]}\n"
            current_data_string += f"penalty: {self.penalty[-1]}\n"
            current_data_string += f"x_next: {self.x_next[-1]}\n"
            current_data_string += f"y_next: {self.y_next[-1]}\n"
        else:
            current_data_string += "No entries recorded yet."
        return current_data_string

# Main CSTR environment class
class CSTR(gym.Env):
    measurement_noise: bool = False
    additive_process_uncertainty: bool = False
    parametric_uncertainty: bool = False

    diff_twice: bool = False

    tight_initialization: bool = False
    terminal_cost_approximation: bool = False
    penalty_weight: float = 10.0

    sb3_mode: bool = False
    sb3_test_mode: bool = False
    sb3_n_ic: int = 1
    sb3_ic_counter: int = 0

    def __init__(
            self,
            seed:int = 1234,
            terminate_on_cv: bool = True,
            max_steps: int = 100,
            max_steps_of_violation:int=3,
            gamma: float = 0.99,
            ) -> None:
        super().__init__()

        # Set environment settings
        self.settings = CSTRSettings(
            seed=seed,
            terminate_on_cv=terminate_on_cv,
            max_steps=max_steps,
            max_steps_of_violation=max_steps_of_violation,
            gamma=gamma,
            )

        # Random number generators for initial state and uncertain parameters
        self.rng = np.random.default_rng(seed)
        self.rng_uncertain_params = np.random.default_rng(seed + 1)
        self.rng_measurements = np.random.default_rng(seed + 2)
        self.std_measurements = np.array([0.025, 0.025, 0.5, 0.5, 0.0, 0.0]).reshape(-1, 1)

        # Define action and observation spaces for RL
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=float)
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(6,), dtype=float)

        # Initialize process, initial, and terminal bounds
        self._init_process_bounds()
        self._init_init_bounds()
        self._init_term_bounds()

        # Setup system matrices and integrator
        self._setup_system()

        # Initialize state, parameters, and time
        self.x_num = self.sample_initial_state()
        self.y_num = self.x_num.copy() 
        self.uncertain_params = self.sample_uncertain_params()
        self.current_step = 0
        self.time = 0.0
        self.dt = 0.005  # Integration time step

        # Initialize history and termination/truncation buffers
        self.data = History(x_init = self.x_num.copy(), y_init = self.y_num.copy(), time = self.time)
        self.last_terminations = deque(maxlen = 5)
        # self.last_terminations = deque(maxlen = 1)
        self.last_truncations = deque(maxlen = self.settings.max_steps_of_violation)

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
        percent_off_bounds = 0.0
        if self.tight_initialization:
            percent_off_bounds = 0.1
        self.init_bounds = ProcessBounds(
            x_lb=np.array([
                0.1 * (1 + percent_off_bounds),
                0.1 * (1 + percent_off_bounds),
                80.0 * (1 + percent_off_bounds),
                80.0 * (1 + percent_off_bounds),
                5.0 * (1 + percent_off_bounds),
                -8500.0 * (1 + percent_off_bounds)
                ]).reshape(-1, 1),
            x_ub=np.array([
                2.0 * (1 - percent_off_bounds),
                2.0 * (1 - percent_off_bounds),
                140 * (1 - percent_off_bounds),
                140 * (1 - percent_off_bounds),
                40.0 * (1 - percent_off_bounds),
                0.0 * (1 - percent_off_bounds)
            ]).reshape(-1, 1),
            u_lb=np.array([5.0, -8500]).reshape(-1, 1),
            u_ub=np.array([40.0, 0.0]).reshape(-1, 1),
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
        self.dstates_dt_func = cd.Function("dstates_dt_func", [states, controls, uncertain_params], [dstates_dt], ["states", "controls", "uncertain_params"], ["dstates_dt"])

        # Algebraic equations for previous controls
        F_prev_eq = F - F_prev
        Q_dot_prev_eq = Q_dot - Q_dot_prev
        alg_eq = cd.vertcat(F_prev_eq, Q_dot_prev_eq)

        # Parameters for integration
        p_integration = cd.vertcat(controls, uncertain_params)
        self.p_integration_func = cd.Function("p_integration_func", [controls, uncertain_params], [p_integration], ["controls", "uncertain_params"], ["p_integration"])
        self.p_integration_inv_func = cd.Function("p_integration_inv_func", [p_integration], [controls, uncertain_params], ["p_integration"], ["controls", "uncertain_params"])

        # DAE dictionary for CasADi integrator
        dae_dict = {
            "x": states,
            "z": alg_states,
            "p": p_integration,
            "ode": dstates_dt,
            "alg": alg_eq,
        }

        # Create CasADi integrator for system simulation
        self.integrator = cd.integrator("CSTR", "collocation", dae_dict, 0.0, 0.005)
        self.integrator = self.integrator.factory("CSTR", ["x0", "z0", "p"], ["xf", "zf", "jac:xf:x0", "jac:xf:z0", "jac:xf:p", "jac:zf:x0", "jac:zf:z0", "jac:zf:p"])

        if self.diff_twice:
            name_in = self.integrator.name_in()
            name_out = self.integrator.name_out()
            for f_in in name_in:
                for f_out in name_out:
                    if not f_out.startswith("jac_"):
                        continue
                    self.integrator = self.integrator.factory(
                        "CSTR",
                        self.integrator.name_in(),
                        [*self.integrator.name_out(), f"jac:{f_out}:{f_in}"]
                    )
        
        # Reward
        reward_sym = 1e-2 * (T_R - 126) ** 2 + 1e-2 *(T_K - 120) ** 2

        R = 1e2 * cd.diag([1e0, 1e0])  # Regularization matrix for the control input
        delta_action = alg_states - controls
        delta_action = delta_action / (self.bounds.u_ub - self.bounds.u_lb)
        reward_sym += delta_action.T @ R @ delta_action
        

        # Constraint violations for states
        weights = np.ones(all_states.shape) * self.penalty_weight
        cv_xlb = cd.fmax(self.bounds.x_lb - all_states, np.zeros(all_states.shape))
        cv_xub = cd.fmax(all_states - self.bounds.x_ub, np.zeros(all_states.shape))
        penalty = weights.T @ (cv_xlb + cv_xub)
        reward_sym += penalty

        # Constraint violations for controls
        weights_u = np.ones(controls.shape) * self.penalty_weight * 10.0
        cv_ulb = cd.fmax((self.bounds.u_lb - controls)/(self.bounds.u_ub - self.bounds.u_lb), np.zeros(controls.shape))
        cv_uub = cd.fmax((controls - self.bounds.u_ub)/(self.bounds.u_ub - self.bounds.u_lb), np.zeros(controls.shape))
        penalty = weights_u.T @ (cv_ulb + cv_uub)
        reward_sym += penalty

        reward_sym *= -1.0

        if self.diff_twice:
            hess_reward_all_states, gradient_reward_all_states = cd.hessian(reward_sym, all_states)
            hess_reward_controls, gradient_reward_controls = cd.hessian(reward_sym, controls)
            jac_jac_reward_all_states_controls = cd.jacobian(gradient_reward_all_states, controls)

            self.reward_func = cd.Function(
                "reward_func",
                [all_states, controls],
                [reward_sym, gradient_reward_all_states, gradient_reward_controls, hess_reward_all_states, hess_reward_controls, jac_jac_reward_all_states_controls],
                ["all_states", "controls"],
                ["reward", "grad:reward:all_states", "grad:reward:controls", "hess:reward:all_states", "hess:reward:controls", "jac_jac:reward:all_states:controls"]
                )
        
        else:
            self.reward_func = cd.Function("reward_func", [all_states, controls], [reward_sym], ["all_states", "controls"], ["reward"])
            self.reward_func = self.reward_func.factory("reward_func", ["all_states", "controls"], ["reward", "grad:reward:all_states", "grad:reward:controls"])
        return

    # Step function for RL environment
    def step(self, action: np.ndarray, scaled_action: bool = True, scale_observation: bool = True) -> tuple[np.ndarray, float, bool, bool, dict]:
        info = {}

        x = self.x_num.copy()
        y = self.y_num.copy()
        action = action.reshape(-1, 1)

        # Scale action from [0,1] to physical bounds
        if scaled_action and self.sb3_mode:
            action = self.bounds.u_lb + (self.bounds.u_ub - self.bounds.u_lb) * action
        elif scaled_action and not self.sb3_mode:
            raise NotImplementedError("Scaling actions is only implemented in sb3_mode.")

        # Integrate system dynamics using CasADi
        C_a, C_b, T_R, T_K, F_prev, Q_dot_prev = self.all_state_inv_func(x)
        states, alg_states = self.state_func(C_a, C_b, T_R, T_K), self.alg_state_func(F_prev, Q_dot_prev)
        integration_params = cd.vertcat(action, self.uncertain_params)

        integration_results = self.integrator(x0 = states, z0 = alg_states, p = integration_params)
        
        x_next, z_next = integration_results["xf"], integration_results["zf"]
        jac_x_next_x0, jac_x_next_z0, jac_z_next_x0, jac_z_next_z0, jac_x_next_p, jac_z_next_p  = integration_results["jac_xf_x0"], integration_results["jac_xf_z0"], integration_results["jac_zf_x0"], integration_results["jac_zf_z0"], integration_results["jac_xf_p"], integration_results["jac_zf_p"]

        if self.diff_twice:
            jac_jac_x_next_x0_x0, jac_jac_x_next_x0_z0, jac_jac_x_next_x0_p = integration_results["jac_jac_xf_x0_x0"], integration_results["jac_jac_xf_x0_z0"], integration_results["jac_jac_xf_x0_p"]
            jac_jac_x_next_z0_x0, jac_jac_x_next_z0_z0, jac_jac_x_next_z0_p = integration_results["jac_jac_xf_z0_x0"], integration_results["jac_jac_xf_z0_z0"], integration_results["jac_jac_xf_z0_p"]
            jac_jac_x_next_p_x0, jac_jac_x_next_p_z0, jac_jac_x_next_p_p = integration_results["jac_jac_xf_p_x0"], integration_results["jac_jac_xf_p_z0"], integration_results["jac_jac_xf_p_p"]

            jac_jac_z_next_x0_x0, jac_jac_z_next_x0_z0, jac_jac_z_next_x0_p = integration_results["jac_jac_zf_x0_x0"], integration_results["jac_jac_zf_x0_z0"], integration_results["jac_jac_zf_x0_p"]
            jac_jac_z_next_z0_x0, jac_jac_z_next_z0_z0, jac_jac_z_next_z0_p = integration_results["jac_jac_zf_z0_x0"], integration_results["jac_jac_zf_z0_z0"], integration_results["jac_jac_zf_z0_p"]
            jac_jac_z_next_p_x0, jac_jac_z_next_p_z0, jac_jac_z_next_p_p = integration_results["jac_jac_zf_p_x0"], integration_results["jac_jac_zf_p_z0"], integration_results["jac_jac_zf_p_p"]

            jac_jac_x_next_x0_x0, jac_jac_x_next_x0_z0, jac_jac_x_next_x0_p = jac_jac_x_next_x0_x0.full(), jac_jac_x_next_x0_z0.full(), jac_jac_x_next_x0_p.full()
            jac_jac_x_next_z0_x0, jac_jac_x_next_z0_z0, jac_jac_x_next_z0_p = jac_jac_x_next_z0_x0.full(), jac_jac_x_next_z0_z0.full(), jac_jac_x_next_z0_p.full()
            jac_jac_x_next_p_x0, jac_jac_x_next_p_z0, jac_jac_x_next_p_p = jac_jac_x_next_p_x0.full(), jac_jac_x_next_p_z0.full(), jac_jac_x_next_p_p.full()

            jac_jac_z_next_x0_x0, jac_jac_z_next_x0_z0, jac_jac_z_next_x0_p = jac_jac_z_next_x0_x0.full(), jac_jac_z_next_x0_z0.full(), jac_jac_z_next_x0_p.full()
            jac_jac_z_next_z0_x0, jac_jac_z_next_z0_z0, jac_jac_z_next_z0_p = jac_jac_z_next_z0_x0.full(), jac_jac_z_next_z0_z0.full(), jac_jac_z_next_z0_p.full()
            jac_jac_z_next_p_x0, jac_jac_z_next_p_z0, jac_jac_z_next_p_p = jac_jac_z_next_p_x0.full(), jac_jac_z_next_p_z0.full(), jac_jac_z_next_p_p.full()

        x_next = x_next.full()
        z_next = z_next.full()

        jac_s_next_s = cd.vertcat(cd.horzcat(jac_x_next_x0, jac_x_next_z0), cd.horzcat(jac_z_next_x0, jac_z_next_z0)).full()

        jac_x_next_a_T, jac_x_next_p_T = self.p_integration_inv_func(jac_x_next_p.T)
        jac_z_next_a_T, jac_z_next_p_T = self.p_integration_inv_func(jac_z_next_p.T)

        jac_x_next_a = jac_x_next_a_T.T
        jac_z_next_a = jac_z_next_a_T.T
        jac_s_next_a = cd.vertcat(jac_x_next_a, jac_z_next_a).full()

        if self.diff_twice:
            n_x = self.x_num.shape[0] - action.shape[0]
            n_z = action.shape[0]
            n_a = action.shape[0]
            n_p = self.uncertain_params.shape[0]

            jac_jac_x_next_x0_x0 = jac_jac_x_next_x0_x0.reshape(n_x, n_x, n_x, order = "F")
            jac_jac_x_next_x0_z0 = jac_jac_x_next_x0_z0.reshape(n_x, n_x, n_z, order = "F")
            jac_jac_x_next_x0_p = jac_jac_x_next_x0_p.reshape(n_x, n_x, n_a + n_p, order = "F")

            jac_jac_x_next_z0_x0 = jac_jac_x_next_z0_x0.reshape(n_x, n_z, n_x, order = "F")
            jac_jac_x_next_z0_z0 = jac_jac_x_next_z0_z0.reshape(n_x, n_z, n_z, order = "F")
            jac_jac_x_next_z0_p = jac_jac_x_next_z0_p.reshape(n_x, n_z, n_a + n_p, order = "F")

            jac_jac_x_next_p_x0 = jac_jac_x_next_p_x0.reshape(n_x, n_a + n_p, n_x, order = "F")
            jac_jac_x_next_p_z0 = jac_jac_x_next_p_z0.reshape(n_x, n_a + n_p, n_z, order = "F")
            jac_jac_x_next_p_p = jac_jac_x_next_p_p.reshape(n_x, n_a + n_p, n_a + n_p, order = "F")

            jac_jac_z_next_x0_x0 = jac_jac_z_next_x0_x0.reshape(n_z, n_x, n_x, order = "F")
            jac_jac_z_next_x0_z0 = jac_jac_z_next_x0_z0.reshape(n_z, n_x, n_z, order = "F")
            jac_jac_z_next_x0_p = jac_jac_z_next_x0_p.reshape(n_z, n_x, n_a + n_p, order = "F")

            jac_jac_z_next_z0_x0 = jac_jac_z_next_z0_x0.reshape(n_z, n_z, n_x, order = "F")
            jac_jac_z_next_z0_z0 = jac_jac_z_next_z0_z0.reshape(n_z, n_z, n_z, order = "F")
            jac_jac_z_next_z0_p = jac_jac_z_next_z0_p.reshape(n_z, n_z, n_a + n_p, order = "F")

            jac_jac_z_next_p_x0 = jac_jac_z_next_p_x0.reshape(n_z, n_a + n_p, n_x, order = "F")
            jac_jac_z_next_p_z0 = jac_jac_z_next_p_z0.reshape(n_z, n_a + n_p, n_z, order = "F")
            jac_jac_z_next_p_p = jac_jac_z_next_p_p.reshape(n_z, n_a + n_p, n_a + n_p, order = "F")

            # Stack all x0 and z0 related second derivatives. We go from 2x 9 = 18 to only 1 x 9 (but larger)
            jac_jac_s_next_x0_x0 = np.concatenate((jac_jac_x_next_x0_x0, jac_jac_z_next_x0_x0), axis = 0)
            jac_jac_s_next_x0_z0 = np.concatenate((jac_jac_x_next_x0_z0, jac_jac_z_next_x0_z0), axis = 0)
            jac_jac_s_next_x0_p = np.concatenate((jac_jac_x_next_x0_p, jac_jac_z_next_x0_p), axis = 0)

            jac_jac_s_next_z0_x0 = np.concatenate((jac_jac_x_next_z0_x0, jac_jac_z_next_z0_x0), axis = 0)
            jac_jac_s_next_z0_z0 = np.concatenate((jac_jac_x_next_z0_z0, jac_jac_z_next_z0_z0), axis = 0)
            jac_jac_s_next_z0_p = np.concatenate((jac_jac_x_next_z0_p, jac_jac_z_next_z0_p), axis = 0)

            jac_jac_s_next_p_x0 = np.concatenate((jac_jac_x_next_p_x0, jac_jac_z_next_p_x0), axis = 0)
            jac_jac_s_next_p_z0 = np.concatenate((jac_jac_x_next_p_z0, jac_jac_z_next_p_z0), axis = 0)
            jac_jac_s_next_p_p = np.concatenate((jac_jac_x_next_p_p, jac_jac_z_next_p_p), axis = 0)

            # Now stack all the second derivatives into combined second derivative matrices
            jac_jac_s_next_s_x0 = np.concatenate((jac_jac_s_next_x0_x0, jac_jac_s_next_z0_x0), axis = 1)
            jac_jac_s_next_s_z0 = np.concatenate((jac_jac_s_next_x0_z0, jac_jac_s_next_z0_z0), axis = 1)
            jac_jac_s_next_s_p  = np.concatenate((jac_jac_s_next_x0_p, jac_jac_s_next_z0_p), axis = 1)

            jac_jac_s_next_s_s = np.concatenate((jac_jac_s_next_s_x0, jac_jac_s_next_s_z0), axis = -1)
            jac_jac_s_next_p_s = np.concatenate((jac_jac_s_next_p_x0, jac_jac_s_next_p_z0), axis = -1)

            # Now extract all a from p related second derivatives
            jac_jac_s_next_s_a = jac_jac_s_next_s_p[:, :, :n_a]
            jac_jac_s_next_a_a = jac_jac_s_next_p_p[:, :n_a, :n_a]

            pass

        # Check for termination/truncation conditions
        local_termination, local_truncation, time_exceeded = self._local_termination_truncation_check(x, action)
        self.last_terminations.append(local_termination)
        self.last_truncations.append(local_truncation)
        terminated, truncated = self._termination_truncation_check()
        if time_exceeded:
            terminated = True

        info["time_exceeded"] = time_exceeded
        info["terminated"] = terminated
        info["truncated"] = truncated

        # Compute reward, stage cost, and penalty
        if not self.diff_twice:
            reward, grad_r_s, grad_r_a, stage_cost, penalty = self._get_reward(y, action)
        else:
            reward, grad_r_s, grad_r_a, hess_r_s, hess_r_a, jac_jac_r_sa, stage_cost, penalty = self._get_reward(y, action)

        if self.terminal_cost_approximation:
            if time_exceeded or terminated:
                reward *=  1 / (1 - self.settings.gamma) # Penalty for reaching max time (assuming constant offset)
                grad_r_s *= 1 / (1 - self.settings.gamma)
                grad_r_a *= 1 / (1 - self.settings.gamma)
                if self.diff_twice:
                    hess_r_s *= 1 / (1 - self.settings.gamma)
                    hess_r_a *= 1 / (1 - self.settings.gamma)
                    jac_jac_r_sa *= 1 / (1 - self.settings.gamma)

        # Update state
        if self.additive_process_uncertainty and not self.measurement_noise:
            measurement_noise = self.rng_measurements.normal(loc = np.zeros(shape = self.x_num.shape), scale = self.std_measurements)
            self.x_num = observation = np.vstack((x_next, z_next)).copy() + measurement_noise
            self.y_num = observation.copy()
        elif self.measurement_noise and not self.additive_process_uncertainty:
            measurement_noise = self.rng_measurements.normal(loc = np.zeros(shape = self.x_num.shape), scale = self.std_measurements)
            self.x_num = observation = np.vstack((x_next, z_next)).copy()
            self.y_num = observation.copy() + measurement_noise
        elif self.additive_process_uncertainty and self.measurement_noise:
            measurement_noise = self.rng_measurements.normal(loc = np.zeros(shape = self.x_num.shape), scale = self.std_measurements)
            self.x_num = observation = np.vstack((x_next, z_next)).copy() + measurement_noise
            self.y_num = observation.copy() + measurement_noise
        else:
            self.x_num = observation = np.vstack((x_next, z_next)).copy()
            self.y_num = observation.copy()

        # Store transition in history
        self.data.remember(
            x = x,
            y = y,
            u = action,
            time = self.time,
            r = reward,
            stage_cost = stage_cost,
            penalty = penalty,
            x_next = self.x_num,
            y_next = observation,
            )

        # Scale observation to [0,1] if required
        if scale_observation and self.sb3_mode:
            observation = (observation - self.bounds.x_lb) / (self.bounds.x_ub - self.bounds.x_lb)
        elif scale_observation and not self.sb3_mode:
            raise NotImplementedError("Scaling observations is only implemented in sb3_mode.")

        self.current_step += 1
        self.time += self.dt

        if self.parametric_uncertainty:
            self.uncertain_params = self.sample_uncertain_params()

        if not self.diff_twice and not self.sb3_mode:
            return observation.T, jac_s_next_s, jac_s_next_a, reward, grad_r_s, grad_r_a, terminated, truncated, info
        elif self.diff_twice and not self.sb3_mode:
            returned_items = (
                observation.T,
                jac_s_next_s,
                jac_s_next_a,
                jac_jac_s_next_s_s,
                jac_jac_s_next_s_a,
                jac_jac_s_next_a_a,
                reward,
                grad_r_s,
                grad_r_a,
                hess_r_s,
                jac_jac_r_sa,
                hess_r_a,
                terminated,
                truncated,
                info
            )
        elif self.sb3_mode:
            returned_items = (
                observation.T,
                reward,
                terminated,
                truncated,
                info
            )
        else:
            raise NotImplementedError("This combination of settings is not implemented. You have diff_twice =", self.diff_twice, "and sb3_mode =", self.sb3_mode)
        return returned_items
        

    # Compute reward, stage cost, and penalty for RL
    def _get_reward(self, state: np.ndarray, action: np.ndarray) -> tuple[float, float, float]:
        C_a, C_b, T_R, T_K, F_prev, Q_dot_prev = self.all_state_inv_func(state)
        F, Q_dot = self.control_inv_func(action)
        stage_cost = 1e-2 * (T_R - 126) ** 2 + 1e-2 *(T_K - 120) ** 2

        R = 1e2 * cd.diag([1e0, 1e0])
        delta_action = state[-2:, :] - action
        delta_action = delta_action / (self.bounds.u_ub - self.bounds.u_lb)
        stage_cost += delta_action.T @ R @ delta_action
        

        # Constraint violations for states
        weights = np.ones(state.shape) * self.penalty_weight
        cv_xlb = np.max([self.bounds.x_lb - state, np.zeros(state.shape)], axis = 0)
        cv_xub = np.max([state - self.bounds.x_ub, np.zeros(state.shape)], axis = 0)
        penalty = weights.T @ (cv_xlb + cv_xub)
        penalty = penalty[0, 0]

        # Constraint violations for controls
        weights_u = np.ones(action.shape) * self.penalty_weight * 10.0
        cv_ulb = np.max([(self.bounds.u_lb - action)/(self.bounds.u_ub - self.bounds.u_lb), np.zeros(action.shape)], axis = 0)
        cv_uub = np.max([(action - self.bounds.u_ub)/(self.bounds.u_ub - self.bounds.u_lb), np.zeros(action.shape)], axis = 0)
        penalty += weights_u.T @ (cv_ulb + cv_uub)
        penalty = penalty[0, 0]

        if not self.diff_twice:
            reward, grad_reward_state, grad_reward_action = self.reward_func(state, action)
            return reward.full(), grad_reward_state.full(), grad_reward_action.full(), stage_cost, penalty
        else:
            reward, grad_reward_state, grad_reward_action, hess_reward_state, hess_reward_action, jac_jac_reward_state_action = self.reward_func(state, action)
            return reward.full(), grad_reward_state.full(), grad_reward_action.full(), hess_reward_state.full(), hess_reward_action.full(), jac_jac_reward_state_action.full(), stage_cost, penalty
    

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
        
        return terminated, truncated, time_exceeded

    
    # Check for overall termination/truncation based on history
    def _termination_truncation_check(self) -> tuple[bool, bool]:
        terminated = False
        truncated = False
        return terminated, truncated

    def get_full_env_state(self, scale_observation: bool = True) -> np.ndarray:
        state = self.x_num.copy()
        observation = self.y_num.copy()
        if scale_observation:
            state = (state - self.bounds.x_lb) / (self.bounds.x_ub - self.bounds.x_lb)
            observation = (observation - self.bounds.x_lb) / (self.bounds.x_ub - self.bounds.x_lb)
        return state.T, observation.T, self.uncertain_params.T, self.current_step, self.time

    # Set observation (state) manually, with optional scaling
    def copy_full_env_state(self, full_env_state: tuple[np.ndarray, np.ndarray, np.ndarray, int, float], scale_observation: bool = True) -> np.ndarray:
        state, observation, uncertain_params, current_step, time  = full_env_state
        if scale_observation:
            state = self.bounds.x_lb + (self.bounds.x_ub - self.bounds.x_lb) * state
            observation = self.bounds.x_lb + (self.bounds.x_ub - self.bounds.x_lb) * observation

        if observation.ndim == 1:
            if observation.shape[0] == 6:
                observation = observation.reshape(-1, 1)

        elif observation.ndim == 2:
            if observation.shape[1] == 6 and observation.shape[0] == 1:
                observation = observation.T

        else:
            raise ValueError(f"Observation must have shape (6,), (1, 6), or (6, 1). You have provided shape {observation.shape}.")

        if state.ndim == 1:
            if state.shape[0] == 6:
                state = state.reshape(-1, 1)
        
        elif state.ndim == 2:
            if state.shape[1] == 6 and state.shape[0] == 1:
                state = state.T

        else:
            raise ValueError(f"State must have shape (6,), (1, 6), or (6, 1). You have provided shape {state.shape}.")

        self.x_num = state.copy()
        self.y_num = observation.copy()
        self.uncertain_params = uncertain_params.T.copy()
        self.current_step = current_step
        self.time = time
        return state.T, observation.T, uncertain_params

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

        if self.sb3_mode and self.settings.seed is not None:
            if self.sb3_ic_counter == self.sb3_n_ic:
                self.sb3_ic_counter = 0
            if not self.sb3_test_mode:
                seed = self.settings.seed + self.sb3_ic_counter
                self.sb3_ic_counter += 1
            else:
                pass         

        # Update RNGs if new seed is provided
        if not self.sb3_mode and seed is not None:
            self.settings.seed = seed
            self.rng = np.random.default_rng(seed)
            self.rng_uncertain_params = np.random.default_rng(seed + 1)
            self.rng_measurements = np.random.default_rng(seed + 2)
        elif self.sb3_mode and self.settings.seed is not None:
            if self.sb3_test_mode:
                self.settings.seed = seed
            self.rng = np.random.default_rng(seed)
            self.rng_uncertain_params = np.random.default_rng(seed + 1)
            self.rng_measurements = np.random.default_rng(seed + 2)

        # Sample initial state and parameters
        self.x_num = self.sample_initial_state()
        self.uncertain_params = self.sample_uncertain_params()
        self.current_step = 0
        self.time = 0.0




        observation = self.x_num.copy()
        measurement_noise = self.rng_measurements.normal(loc = np.zeros(observation.shape), scale = self.std_measurements)
        observation += measurement_noise
        self.y_num = observation.copy()

        # Reset history and termination/truncation buffers
        self.data = History(x_init= self.x_num.copy(), y_init = self.y_num.copy(), time = self.time)
        self.last_terminations = deque(maxlen = self.last_terminations.maxlen)
        self.last_truncations = deque(maxlen = self.settings.max_steps_of_violation)


        if scale_observation:
            observation = (observation - self.bounds.x_lb) / (self.bounds.x_ub - self.bounds.x_lb)



        return observation.T, info
    
    def plot(self, filepath: str = None):

        if not self.data.flags["compactified"]:
            self.data.compactify()

        time_data = self.data.time

        cA_data = self.data.x[:, 0, 0]
        cB_data = self.data.x[:, 1, 0]
        TR_data = self.data.x[:, 2, 0]
        TK_data = self.data.x[:, 3, 0]

        cA_meas_data = self.data.y[:, 0, 0]
        cB_meas_data = self.data.y[:, 1, 0]
        TR_meas_data = self.data.y[:, 2, 0]
        TK_meas_data = self.data.y[:, 3, 0]

        F_data = self.data.u[:, 0, 0]
        Q_data = self.data.u[:, 1, 0] * 1e-3

        ncols = 3
        nrows = 2
        scaling_factor = 1.0
        figsize = (5 * ncols * scaling_factor, 4 * nrows * scaling_factor)

        alpha_meas = 0.5
        alpha_const = 0.25

        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(ncols = ncols, nrows = nrows, sharex = True, figsize = figsize, constrained_layout = True)

        # Concentration A C_A
        ax[0, 0].plot(time_data, cA_data, label = "True system", color = "tab:blue", alpha = alpha_meas)
        ax[0, 0].plot(time_data, cA_meas_data, label = "Measurement", color = "tab:blue", linestyle = "None", marker = "x")
        ax[0, 0].fill_between(x = time_data, y1 = np.ones(shape = time_data.shape) * 0.1, y2 = np.ones(shape = time_data.shape) * (- 0.1), color = "red", alpha = alpha_const)
        ax[0, 0].fill_between(x = time_data, y1 = np.ones(shape = time_data.shape) * 2.0, y2 = np.ones(shape = time_data.shape) * (2.2), color = "red", alpha = alpha_const)
        ax[0, 0].set_ylim([0, 2.1])
        ax[0, 0].set_ylabel(r"$C_\mathrm{A} ~ [\mathrm{mol\,L^{-1}}]$")

        # Concentration B C_B
        ax[1, 0].plot(time_data, cB_data, label = "True system", color = "tab:blue", alpha = alpha_meas)
        ax[1, 0].plot(time_data, cB_meas_data, label = "Measurement", color = "tab:blue", linestyle = "None", marker = "x")
        ax[1, 0].fill_between(x = time_data, y1 = np.ones(shape = time_data.shape) * 0.1, y2 = np.ones(shape = time_data.shape) * (- 0.1), color = "red", alpha = alpha_const)
        ax[1, 0].fill_between(x = time_data, y1 = np.ones(shape = time_data.shape) * 2.0, y2 = np.ones(shape = time_data.shape) * (2.2), color = "red", alpha = alpha_const)
        ax[1, 0].set_ylim([0, 2.1])
        ax[1, 0].set_ylabel(r"$C_\mathrm{B} ~ [\mathrm{mol\,L^{-1}}]$")
        ax[1, 0].set_xlabel(r"time $[\mathrm{h}]$")

        # Temperature reactor T_R
        ax[0, 1].plot(time_data, TR_data, label = "True system", color = "tab:blue", alpha = alpha_meas)
        ax[0, 1].plot(time_data, TR_meas_data, label = "Measurement", color = "tab:blue", linestyle = "None", marker = "x")
        ax[0, 1].plot(time_data, np.ones(time_data.shape) * 126.0, label = "Setpoint", linestyle = "dashed", color ="tab:red")
        ax[0, 1].fill_between(x = time_data, y1 = np.ones(shape = time_data.shape) * 80, y2 = np.ones(shape = time_data.shape) * (70), color = "red", alpha = alpha_const)
        ax[0, 1].fill_between(x = time_data, y1 = np.ones(shape = time_data.shape) * 140, y2 = np.ones(shape = time_data.shape) * (150), color = "red", alpha = alpha_const)
        ax[0, 1].set_ylim([75, 145])
        ax[0, 1].set_ylabel(r"$T_\mathrm{R} ~  [\mathrm{^\circ C}]$")

        # Temperature Coolant T_K
        ax[1, 1].plot(time_data, TK_data, label = "True system", color = "tab:blue", alpha = alpha_meas)
        ax[1, 1].plot(time_data, TK_meas_data, label = "Measurement", color = "tab:blue", linestyle = "None", marker = "x")
        ax[1, 1].plot(time_data, np.ones(time_data.shape) * 120.0, label = "Setpoint", linestyle = "dashed", color ="tab:red")
        ax[1, 1].fill_between(x = time_data, y1 = np.ones(shape = time_data.shape) * 80, y2 = np.ones(shape = time_data.shape) * 70, color = "red", alpha = alpha_const)
        ax[1, 1].fill_between(x = time_data, y1 = np.ones(shape = time_data.shape) * 140, y2 = np.ones(shape = time_data.shape) * 150, color = "red", alpha = alpha_const)
        ax[1, 1].set_ylim([75, 145])
        ax[1, 1].set_ylabel(r"$T_\mathrm{K} ~ [\mathrm{^\circ C}]$")
        ax[1, 1].set_xlabel(r"time $[\mathrm{h}]$")

        # Dilution rate F
        ax[0, 2].step(time_data, F_data, label = "True system", color = "tab:blue", where = "pre")
        ax[0, 2].fill_between(x = time_data, y1 = np.ones(shape = time_data.shape) * 5, y2 = np.ones(shape = time_data.shape) * -5, color = "red", alpha = alpha_const)
        ax[0, 2].fill_between(x = time_data, y1 = np.ones(shape = time_data.shape) * 40, y2 = np.ones(shape = time_data.shape) * 50, color = "red", alpha = alpha_const)
        ax[0, 2].set_ylim([2.5, 42.5])
        ax[0, 2].set_ylabel(r"$F ~  [\mathrm{h^{-1}}]$")

        # Cooling rate Q
        ax[1, 2].step(time_data, Q_data, label = "True system", color = "tab:blue", where = "pre")
        ax[1, 2].fill_between(x = time_data, y1 = np.ones(shape = time_data.shape) * 0., y2 = np.ones(shape = time_data.shape) * 1, color = "red", alpha = alpha_const)
        ax[1, 2].fill_between(x = time_data, y1 = np.ones(shape = time_data.shape) * -8.5, y2 = np.ones(shape = time_data.shape) * -10, color = "red", alpha = alpha_const)
        ax[1, 2].set_ylim([-9, 0.5])
        ax[1, 2].set_ylabel(r"$Q ~ [\mathrm{kJ \, h^{-1}}]$")
        ax[1, 2].set_xlabel(r"time $[\mathrm{h}]$")


        ax.flatten()[0].set_xlim([time_data[0], time_data[-1]])
        for axis in ax.flatten():
            axis.grid()
            axis.legend(loc = "lower right")

        if filepath is None:
            plt.show()
        else:
            plt.savefig(filepath, dpi = 1200.0)
        return

# Example usage and test
if __name__ == "__main__":
    env = CSTR(seed=1234, terminate_on_cv=True, max_steps=100, max_steps_of_violation=5)
    obs, info = env.reset()

    # observation, reward, terminated, truncated, info = env.step(np.array([0.5, 0.5]), scaled_action=True, scale_observation=False)
    observation, reward, terminated, truncated, info = env.step(np.array([0.5, 0.5]), scaled_action=True, scale_observation=True)

    for idx in range(50):
        observation, reward, terminated, truncated, info = env.step(np.array([0.5, 0.5]), scaled_action=True, scale_observation=True)

    env.plot()
    import os
    env.plot(os.path.join("CSTR", "data", "test_trajectory.png"))