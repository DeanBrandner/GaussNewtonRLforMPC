import casadi as cd
import numpy as np
from RL_MPC import RL_MPC
from do_mpc.differentiator.helper import  NLPDifferentiatorStatus
from dataclasses import dataclass, field


def tensor_vector_product(tensor: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Perform a tensor-vector product.

    Args:
        tensor (np.ndarray): A 3D tensor of shape (n, m, m).
        vector (np.ndarray): A 2D vector of shape (m, 1).

    Returns:
        np.ndarray: The result of the tensor-vector product, of shape (n, m).
    """
    if len(vector.shape) == 1:
        vector = np.expand_dims(vector, axis=-1)
    
    # Expand dimensions of vector to match tensor for broadcasting
    vector = np.expand_dims(vector, axis=-1)
    
    # Perform element-wise multiplication and sum over the action dimension
    result = np.sum(tensor * vector, axis=1)
    return result


class Noise_generator:
    """
    Generates noise for exploration in RL agents.

    Args:
        shape (tuple): Shape of the noise array.
        noise_type (str): Type of noise, "normal" or "uniform".
        noise_std (np.ndarray or float): Standard deviation or range for the noise.
        seed (int): Random seed for reproducibility.
    """
    def __init__(self, shape: tuple, noise_type: str = "normal", noise_std: np.ndarray = 1e-6, seed: int = None):
        self.shape = shape
        self.seed = seed
        self.noise_type = noise_type
        
        # Initialize random number generator with the given seed
        self.noise_rng = np.random.default_rng(seed)

        # Ensure noise_std has the correct shape
        if isinstance(noise_std, float):
            noise_std = np.ones(shape) * noise_std
        elif len(noise_std.shape) == 1 and noise_std.shape[0] == shape[0]:
            noise_std = noise_std.reshape(-1, 1)
        elif noise_std.shape != shape:
            raise ValueError("The shape of the noise_std must be the same as the shape of the noise.")
        
        self.noise_std = noise_std

    def __call__(self):
        """
        Generate noise according to the specified type and standard deviation.
        Returns:
            np.ndarray: Noise array.
        """
        if self.noise_type == "normal":
            noise = self.noise_rng.normal(loc=np.zeros(self.shape), scale=self.noise_std)
        elif self.noise_type == "uniform":
            noise = self.noise_rng.uniform(low=-self.noise_std, high=self.noise_std)
        else:
            raise ValueError("Unknown noise type: {}. Supported types are 'normal' and 'uniform'.".format(self.noise_type))
        return noise
    
    def reset(self, seed: int = None):
        """
        Reset the random number generator with a new seed.

        Args:
            seed (int): New random seed.
        """
        if not seed is None:
            self.seed = seed
        self.noise_rng = np.random.default_rng(self.seed)
        
### Helper
@dataclass
class NLPDifferentiatorSettings_RL():
    """Settings for NLPDifferentiator.
    """

    lin_solver: str = field(default_factory = lambda: 'casadi')
    """
    Choose the linear solver for the KKT system. 
    Can be ``'casadi'``, ``'scipy'`` or ``'lstsq'`` (least squares).
    """

    check_LICQ: bool = True
    """
    Check if the constraints are linearly independent at the given optimal solution.    
    The result of this check is stored in :py:class:`NLPDifferentiatorStatus`.

    Warning:
        This feature is computationally demanding and should only be used for debugging purposes.
    """

    check_SC: bool = True
    """
    Check if strict complementarity holds.  
    The result of this check is stored in :py:class:`NLPDifferentiatorStatus`.
    """

    track_residuals: bool = True
    """
    Compute the residuals of the KKT system.
    """

    check_rank: bool = False
    """
    Check if the KKT matrix has full rank.
    The result of this check is stored in :py:class:`NLPDifferentiatorStatus`.
    
    Warning:
        This feature is computationally demanding and should only be used for debugging purposes.
    """

    lstsq_fallback: bool = False
    """
    Fallback to least squares if the linear solver fails.
    """

    active_set_tol : float = 1e-6
    """
    Tolerance for the active set constraints. 
    """

    set_lam_zero: bool = False
    """
    Set the Lagrangen multipliers to exactly zero if they are below the tolerance.
    """

    recover_LICQ: bool = True
    """
    Recover the LICQ if it is not satisfied.
    """

    enforce_symmetric_hessian: bool = False
    """
    Enforce the Hessians to be symmetric.
    """

    enable_fd: bool = False
    """
    Enable finite differences for sensitivity computation
    """

    fd_eps: float = np.sqrt(np.finfo(float).eps)
    """
    Finite difference step size.
    """

    second_order: bool = False
    """
    Enable second order sensitivity computation.
    """

    reduce_p: bool = False
    """
    Reduce the parameter vector to the parameters that are actually used in the NLP.
    """

class NLP_differentiator:

    def __init__(self, mpc: RL_MPC = None, **kwargs):
        
        self.mpc = mpc

        self.nlp, self.nlp_bounds = self._get_do_mpc_nlp()

        self._status = NLPDifferentiatorStatus()
        self._settings = NLPDifferentiatorSettings_RL(**kwargs)
        
        self.x_scaling_factors = mpc.opt_x_scaling.master
        self.u_scaling_factors = mpc._u_scaling.master

        

        opt_x = mpc.opt_x.master
        _u0 = mpc.opt_x["_u", 0, 0]
        self._u0_extractor = cd.Function("u0_extractor", [opt_x], [_u0], ["opt_x"], ["u0"])


        opt_p = mpc.opt_p.master
        _p = mpc.opt_p["_p", 0]
        self._p_extractor = cd.Function("p_extractor", [opt_p], [_p], ["opt_p"], ["p"])

        self._prepare_differentiator()

        if self.status.reduced_nlp:
            opt_x = opt_x[self.det_sym_idx_dict["opt_x"]]
        self._u0_extractor = cd.Function("u0_extractor", [opt_x], [_u0], ["opt_x"], ["u0"])

        return
    

    @property
    def status(self) -> NLPDifferentiatorStatus:
        """
        Status of the NLP differentiator. This is an annotated dataclass that can also be printed for convenience.
        See :py:class:`do_mpc.differentiator.helper.NLPDifferentiatorStatus` for more information.
        """
        return self._status
    

    @property
    def settings(self) -> NLPDifferentiatorSettings_RL:
        """
        Settings of the NLP differentiator. This is an annotated dataclass that can also be printed for convenience.
        See :py:class:`do_mpc.differentiator.helper.NLPDifferentiatorSettings` for more information.

        **Example**:

        ::

            nlp_diff = NLPDifferentiator(nlp, nlp_bounds)
            nlp_diff.settings.check_licq = False

        Note:
            Settings can also be passed as keyword arguments to the constructor of :py:class:`NLPDifferentiator`.
        """
        return self._settings
    


    def jac_action_parameters(self, mpc):
        if not isinstance(mpc, dict):
            nlp_sol = self._get_do_mpc_nlp_sol(mpc)
        else:
            nlp_sol = mpc.copy()

        p_num = nlp_sol["p"]
        p_RL_num = self._p_extractor(p_num)

        nlp_sol_mandatory_keys = ['x', 'lam_g', 'lam_x', 'g']
        
        if not isinstance(nlp_sol, dict):
            raise ValueError('nlp_sol must be a dictionary.')

        if not set(nlp_sol.keys()).issuperset(set(nlp_sol_mandatory_keys)):
            raise ValueError('nlp_sol must contain keys {}.'.format(nlp_sol_mandatory_keys))

        if isinstance(p_num, (float, int)):
            p_num = cd.DM(p_num)
        elif isinstance(p_num, np.ndarray):
            p_num = cd.DM(p_num)
        elif isinstance(p_num, cd.DM):
            pass
        else:
            raise ValueError('p_num must be a float, int, np.ndarray or DM object. You have {}'.format(type(p_num)))
        
        # reduce NLP solution if necessary
        if self.status.reduced_nlp:
            nlp_sol, p_num = self._reduce_nlp_solution_to_determined(nlp_sol, p_num)

        if not p_RL_num.shape == (self.n_p, 1):
            raise ValueError('p_num must have length {}.'.format(self.n_p))




        # First order sensitivities
        param_sens = self._calculate_sensitivities(nlp_sol, p_num)

        dx_dp_num = self._opt_x_extractor(param_sens)

        if self.status.reduced_nlp:
            pass

        jac_action_parameters = self._u0_extractor(dx_dp_num)
        
        jac_action_parameters = jac_action_parameters * self.u_scaling_factors.full().reshape(self.n_u, 1)
        jac_action_parameters = jac_action_parameters.full()

        return jac_action_parameters
    
    def jac_jac_action_parameters_parameters(self, mpc):
        if not isinstance(mpc, dict):
            nlp_sol = self._get_do_mpc_nlp_sol(mpc)
        else:
            nlp_sol = mpc.copy()

        p_num = nlp_sol["p"]
        p_RL_num = self._p_extractor(p_num)

        nlp_sol_mandatory_keys = ['x', 'lam_g', 'lam_x', 'g']
        
        if not isinstance(nlp_sol, dict):
            raise ValueError('nlp_sol must be a dictionary.')

        if not set(nlp_sol.keys()).issuperset(set(nlp_sol_mandatory_keys)):
            raise ValueError('nlp_sol must contain keys {}.'.format(nlp_sol_mandatory_keys))

        if isinstance(p_num, (float, int)):
            p_num = cd.DM(p_num)
        elif isinstance(p_num, np.ndarray):
            p_num = cd.DM(p_num)
        elif isinstance(p_num, cd.DM):
            pass
        else:
            raise ValueError('p_num must be a float, int, np.ndarray or DM object. You have {}'.format(type(p_num)))
        
        # reduce NLP solution if necessary
        if self.status.reduced_nlp:
            nlp_sol, p_num = self._reduce_nlp_solution_to_determined(nlp_sol, p_num)

        if not p_RL_num.shape == (self.n_p, 1):
            raise ValueError('p_RL_num must have length {}. You have {}'.format(self.n_p, p_RL_num.shape[0]))




        # Calculate first and second order sensitivities
        param_sens, param_sens_second_order = self._calculate_second_order_sensitivities(nlp_sol, p_num)

        # First extract the primal variables
        dx_dp_num = self._opt_x_extractor(param_sens)
        dx2_dp2_num = self._opt_x_extractor(param_sens_second_order)

        # Extract the optimal control action
        jac_action_parameters = self._u0_extractor(dx_dp_num)
        jac_jac_action_parameters_parameters = self._u0_extractor(dx2_dp2_num)

        # Handle the dimensions of jac_jac_action_parameters_parameters
        jac_jac_action_parameters_parameters = self._make_symmetric_tensor(jac_jac_action_parameters_parameters)
        
        jac_action_parameters = jac_action_parameters * self.u_scaling_factors.full().reshape(self.n_u, 1)
        jac_action_parameters = jac_action_parameters.full()
        jac_jac_action_parameters_parameters = jac_jac_action_parameters_parameters * self.u_scaling_factors.full().reshape(self.n_u, 1, 1)

        return jac_action_parameters, jac_jac_action_parameters_parameters
    
    def _make_symmetric_tensor(self, jac_jac_action_parameters_parameters: cd.DM):

        if isinstance(jac_jac_action_parameters_parameters, cd.DM):
            jac_jac_action_parameters_parameters = jac_jac_action_parameters_parameters.full()
        
        jac_tensor = np.zeros((self.n_u, self.n_p, self.n_p))

        for idx in range(self.n_p):
            jac_tensor[:, :, idx] = jac_jac_action_parameters_parameters[:, idx*self.n_p:self.n_p*(idx + 1)]

        return jac_tensor

    def _calculate_sensitivities(self, nlp_sol, p_num):
              
        A_num, B_num = self._get_sensitivity_matrices(nlp_sol, p_num)
        param_sens = self._solve_linear_system(A_num, B_num, lin_solver=self.settings.lin_solver)
        return param_sens
    
    def _calculate_second_order_sensitivities(self, nlp_sol, p_num):
        A_num, B_num = self._get_sensitivity_matrices(nlp_sol, p_num)

        param_sens = self._solve_linear_system(A_num, B_num, lin_solver=self.settings.lin_solver)

        C_num = self._get_second_order_sensitivity_matrices(nlp_sol, p_num, param_sens)

        param_sens_2 = self._solve_linear_system(A_num, C_num, lin_solver=self.settings.lin_solver)

        return param_sens, param_sens_2

    @staticmethod
    def _get_do_mpc_nlp_sol(mpc_object):
        nlp_sol = {}
        nlp_sol["x"] = cd.vertcat(mpc_object.opt_x_num)
        nlp_sol["x_unscaled"] = cd.vertcat(mpc_object.opt_x_num_unscaled)
        nlp_sol["g"] = cd.vertcat(mpc_object.opt_g_num)
        nlp_sol["lam_g"] = cd.vertcat(mpc_object.lam_g_num)
        nlp_sol["lam_x"] = cd.vertcat(mpc_object.lam_x_num)
        nlp_sol["p"] = cd.vertcat(mpc_object.opt_p_num)
        return nlp_sol
        
    def _get_sensitivity_matrices(self, nlp_sol: dict, p_num: cd.DM):

        x_num = nlp_sol["x"]
        lam_x_num = nlp_sol["lam_x"]
        lam_g_num = nlp_sol["lam_g"]

        A_num = self.A_func(x_num, lam_x_num, lam_g_num, p_num)
        B_num = self.B_func(x_num, lam_x_num, lam_g_num, p_num)
        return A_num, B_num
    
    def _get_second_order_sensitivity_matrices(self, nlp_sol: dict, p_num: cd.DM, first_order_sensitivities: cd.DM):
        x_num = nlp_sol["x"]
        lam_x = nlp_sol["lam_x"]
        lam_g = nlp_sol["lam_g"]

        z_num = cd.vertcat(x_num, lam_x, lam_g)

        C_num = self.C_func(z_num, p_num, first_order_sensitivities)

        return C_num
    
    def _solve_linear_system(self, A_num: cd.DM, B_num: cd.DM, lin_solver: str = "mumps") -> cd.DM:
        
        try:
            Linalg_solver = cd.Linsol("Diff", "qr", A_num.sparsity())
            param_sens = Linalg_solver.solve(A_num, -B_num)
        except RuntimeError:
            Linalg_solver = cd.Linsol("Diff", "lapacklu", A_num.sparsity())
            param_sens = Linalg_solver.solve(A_num, -B_num)
        return param_sens
    
    def _reduce_nlp_solution_to_determined(self, nlp_sol: dict, p_num: cd.DM):
        """
        Warning:
            Not part of the public API.
        
        Maps the full NLP solutions to the reduced NLP solutions (the determined variables).

        Args:
            nlp_sol: Full NLP solution.
            p_num: Numerical parameter vector.

        Returns:
            Reduced NLP solution ``nlp_sol_red`` and Reduced parameter vector ``p_num``.

        """

        assert self.status.reduced_nlp, "NLP is not reduced."

        # adapt nlp_sol
        nlp_sol_red = nlp_sol.copy()
        nlp_sol_red["x"] = nlp_sol["x"][self.det_sym_idx_dict["opt_x"]]
        nlp_sol_red["lam_x"] = nlp_sol["lam_x"][self.det_sym_idx_dict["opt_x"]]
        if self.settings.reduce_p:
            p_num = p_num[self.det_sym_idx_dict["opt_p"]]
        
        # backwards compatilibity TODO: remove
        if "x_unscaled" in nlp_sol:
            nlp_sol_red["x_unscaled"] = nlp_sol["x_unscaled"][self.det_sym_idx_dict["opt_x"]]

        return nlp_sol_red, p_num
    
    def _prepare_differentiator(self):

        self._remove_unused_sym_vars()
        self._get_size_metrics()
        self._get_Lagrangian_sym()
        self._prepare_sensitivity_matrices()
        return
    
    def _remove_unused_sym_vars(self):
        """
        Warning:
            Not part of the public API.

        Reduces the NLP by removing symbolic variables 
        for x and p that are not contained in the objective function or the constraints.

        """
        # detect undetermined symbolic variables
        undet_opt_x_idx, det_opt_x_idx = self._detect_undetermined_sym_var("x")
        if self.settings.reduce_p:
            undet_opt_p_idx, det_opt_p_idx = self._detect_undetermined_sym_var("p")
        else:
            undet_opt_p_idx, det_opt_p_idx = np.array([], dtype = np.int64), np.arange(self.nlp["p"].shape[0], dtype = np.int64)
        
        # copy nlp and nlp_bounds
        nlp_red = self.nlp.copy()
        nlp_bounds_red = self.nlp_bounds.copy()

        # adapt nlp
        nlp_red["x"] = self.nlp["x"][det_opt_x_idx]
        nlp_red["p"] = self.nlp["p"][det_opt_p_idx]

        # adapt nlp_bounds
        nlp_bounds_red["lbx"] = self.nlp_bounds["lbx"][det_opt_x_idx]
        nlp_bounds_red["ubx"] = self.nlp_bounds["ubx"][det_opt_x_idx]

        det_sym_idx_dict = {"opt_x":det_opt_x_idx, "opt_p":det_opt_p_idx}
        undet_sym_idx_dict = {"opt_x":undet_opt_x_idx, "opt_p":undet_opt_p_idx}

        N_vars_to_remove = len(undet_sym_idx_dict["opt_x"])+len(undet_sym_idx_dict["opt_p"])
        if N_vars_to_remove > 0:
            self.nlp_unreduced, self.nlp_bounds_unreduced = self.nlp, self.nlp_bounds
            self.nlp, self.nlp_bounds = nlp_red, nlp_bounds_red
            self.det_sym_idx_dict, self.undet_sym_idx_dict = det_sym_idx_dict, undet_sym_idx_dict
            self.status.reduced_nlp = True
        else:
            self.status.reduced_nlp = False
            print("NLP formulation does not contain unused variables.")

    def _get_size_metrics(self):
        """
        Warning:
            Not part of the public API.

        Specifies the number of decision variables, nonlinear constraints and parameters of the NLP.
        """
        self.n_x = self.nlp["x"].shape[0]
        self.n_g = self.nlp["g"].shape[0]
        self.n_p = self._p_extractor(self.nlp["p"]).shape[0]
        self.n_u = self._u0_extractor(0).shape[0]

    def _get_sym_lagrange_multipliers(self):
        """
        Warning:
            Not part of the public API.

        Adds symbolic variables for the Lagrange multipliers to the NLP.
        """

        # Check state constraints
        lbx = self.nlp_bounds["lbx"]
        ubx = self.nlp_bounds["ubx"]

        x_equality = np.isclose(lbx, ubx)
        x_inequality = np.logical_not(x_equality)
        self.where_x_equality = np.where(x_equality)[0]
        self.where_x_inequality = np.where(x_inequality)[0]

        x_upper_bounded = np.logical_and(x_inequality, np.logical_not(np.isinf(ubx)))
        x_lower_bounded = np.logical_and(x_inequality, np.logical_not(np.isinf(lbx)))
        x_upper_lower_bounded = np.logical_and(x_upper_bounded, x_lower_bounded)
        x_upper_bounded = np.logical_and(np.logical_not(x_upper_lower_bounded), x_upper_bounded)
        x_lower_bounded = np.logical_and(np.logical_not(x_upper_lower_bounded), x_lower_bounded)
        x_bounded = np.logical_or(np.logical_or(x_upper_bounded, x_lower_bounded), x_upper_lower_bounded)
        x_unbounded = np.logical_and(x_inequality, np.logical_not(x_bounded))

        self.where_x_upper_bounded = np.where(x_upper_bounded)[0]
        self.where_x_lower_bounded = np.where(x_lower_bounded)[0]
        self.where_x_upper_lower_bounded = np.where(x_upper_lower_bounded)[0]
        self.where_x_bounded = np.where(x_bounded)[0]
        self.where_x_unbounded = np.where(x_unbounded)[0]


        # Check g constraints
        lbg = self.nlp_bounds["lbg"]
        ubg = self.nlp_bounds["ubg"]

        g_equality = np.isclose(lbg, ubg)
        g_inequality = np.logical_not(g_equality)
        self.where_g_equality = np.where(g_equality)[0]
        self.where_g_inequality = np.where(g_inequality)[0]

        g_upper_bounded = np.logical_and(g_inequality, np.logical_not(np.isinf(ubg)))
        g_lower_bounded = np.logical_and(g_inequality, np.logical_not(np.isinf(lbg)))
        g_upper_lower_bounded = np.logical_and(g_upper_bounded, g_lower_bounded)
        g_upper_bounded = np.logical_and(np.logical_not(g_upper_lower_bounded), g_upper_bounded)
        g_lower_bounded = np.logical_and(np.logical_not(g_upper_lower_bounded), g_lower_bounded)
        g_bounded = np.logical_or(np.logical_or(g_upper_bounded, g_lower_bounded), g_upper_lower_bounded)
        g_unbounded = np.logical_and(g_inequality, np.logical_not(g_bounded))

        self.where_g_upper_bounded = np.where(g_upper_bounded)[0]
        self.where_g_lower_bounded = np.where(g_lower_bounded)[0]
        self.where_g_upper_lower_bounded = np.where(g_upper_lower_bounded)[0]
        self.where_g_bounded = np.where(g_bounded)[0]
        self.where_g_unbounded = np.where(g_unbounded)[0]                                   




        # Create symbolic variables for the Lagrange multipliers
        self.nlp["lag_x_all"] = lag_x_all = cd.SX.sym("lag_x_all", self.n_x, 1)
        self.nlp["lag_g_all"] = lag_g_all = cd.SX.sym("lag_g_all", self.n_g, 1)

        # Equality constraints
        self.nlp["eta_x"] = lag_x_all[self.where_x_equality]
        self.nlp["eta_g"] = lag_g_all[self.where_g_equality]
        self.nlp["eta"] = cd.vertcat(self.nlp["eta_g"], self.nlp["eta_x"])

        # Inequality constraints
        self.nlp["lam_x_upper_bounded"] = lag_x_all[self.where_x_upper_bounded]
        self.nlp["lam_x_lower_bounded"] = lag_x_all[self.where_x_lower_bounded]
        self.nlp["lam_x_upper_lower_bounded"] = lag_x_all[self.where_x_upper_lower_bounded]
        self.nlp["lam_x_bounded"] = lag_x_all[self.where_x_bounded]
        self.nlp["lam_x_unbounded"] = lag_x_all[self.where_x_unbounded]
        self.nlp["lam_x"] = cd.vertcat(self.nlp["lam_x_bounded"], self.nlp["lam_x_unbounded"])

        self.nlp["lam_g_upper_bounded"] = lag_g_all[self.where_g_upper_bounded]
        self.nlp["lam_g_lower_bounded"] = lag_g_all[self.where_g_lower_bounded]
        self.nlp["lam_g_upper_lower_bounded"] = lag_g_all[self.where_g_upper_lower_bounded]
        self.nlp["lam_g_bounded"] = lag_g_all[self.where_g_bounded]
        self.nlp["lam_g_unbounded"] = lag_g_all[self.where_g_unbounded]
        self.nlp["lam_g"] = cd.vertcat(self.nlp["lam_g_bounded"], self.nlp["lam_g_unbounded"])

        self.nlp["lam_bounded"] = cd.vertcat(self.nlp["lam_g_bounded"], self.nlp["lam_x_bounded"])
        self.nlp["lam_unbounded"] = cd.vertcat(self.nlp["lam_g_unbounded"], self.nlp["lam_x_unbounded"])
        self.nlp["lam"] = cd.vertcat(self.nlp["lam_g"], self.nlp["lam_x"])

        self.lam_x_bounded_extractor = cd.Function("lam_x_bounded_extractor", [lag_x_all], [self.nlp["lam_x_bounded"]], ["lag_x_all"], ["lam_x_bounded"])
        self.lam_g_bounded_extractor = cd.Function("lam_g_bounded_extractor", [lag_g_all], [self.nlp["lam_g_bounded"]], ["lag_g_all"], ["lam_g_bounded"])
    
    def _stack_primal_dual(self):
        """
        Warning:
            Not part of the public API.

        Stacks the primal and dual variables of the NLP.
        """

        self.nlp["z"] = cd.vertcat(self.nlp["x"], self.nlp["eta"], self.nlp["lam_bounded"])
        self.nlp["z_full"] = cd.vertcat(self.nlp["x"], self.nlp["lag_x_all"], self.nlp["lag_g_all"])
        self.n_z = self.nlp["z"].shape[0]
        self.n_z_full = self.nlp["z_full"].shape[0]
        self._opt_x_extractor = cd.Function("opt_x_extractor", [self.nlp["z"]], [self.nlp["x"]], ["z_opt"], ["opt_x"])

    def _get_Lagrangian_sym(self): 
        """
        Warning:
            Not part of the public API.

        Sets the Lagrangian of the NLP for sensitivity calculation.
        Attention: It is not verified, whether the NLP is in standard form. 
        """

        # Check state constraints
        lbx = self.nlp_bounds["lbx"]
        ubx = self.nlp_bounds["ubx"]

        x_equality = np.isclose(lbx, ubx)
        x_inequality = np.logical_not(x_equality)
        self.where_x_equality = np.where(x_equality)[0]
        self.where_x_inequality = np.where(x_inequality)[0]

        x_upper_bounded = np.logical_and(x_inequality, np.logical_not(np.isinf(ubx)))
        x_lower_bounded = np.logical_and(x_inequality, np.logical_not(np.isinf(lbx)))
        x_upper_lower_bounded = np.logical_and(x_upper_bounded, x_lower_bounded)
        x_upper_bounded = np.logical_and(np.logical_not(x_upper_lower_bounded), x_upper_bounded)
        x_lower_bounded = np.logical_and(np.logical_not(x_upper_lower_bounded), x_lower_bounded)
        x_bounded = np.logical_or(np.logical_or(x_upper_bounded, x_lower_bounded), x_upper_lower_bounded)
        x_unbounded = np.logical_and(x_inequality, np.logical_not(x_bounded))

        self.where_x_upper_bounded = np.where(x_upper_bounded)[0]
        self.where_x_lower_bounded = np.where(x_lower_bounded)[0]
        self.where_x_upper_lower_bounded = np.where(x_upper_lower_bounded)[0]
        self.where_x_bounded = np.where(x_bounded)[0]
        self.where_x_unbounded = np.where(x_unbounded)[0]


        # Check g constraints
        lbg = self.nlp_bounds["lbg"]
        ubg = self.nlp_bounds["ubg"]

        g_equality = np.isclose(lbg, ubg)
        g_inequality = np.logical_not(g_equality)
        self.where_g_equality = np.where(g_equality)[0]
        self.where_g_inequality = np.where(g_inequality)[0]

        g_upper_bounded = np.logical_and(g_inequality, np.logical_not(np.isinf(ubg)))
        g_lower_bounded = np.logical_and(g_inequality, np.logical_not(np.isinf(lbg)))
        g_upper_lower_bounded = np.logical_and(g_upper_bounded, g_lower_bounded)
        g_upper_bounded = np.logical_and(np.logical_not(g_upper_lower_bounded), g_upper_bounded)
        g_lower_bounded = np.logical_and(np.logical_not(g_upper_lower_bounded), g_lower_bounded)
        g_bounded = np.logical_or(np.logical_or(g_upper_bounded, g_lower_bounded), g_upper_lower_bounded)
        g_unbounded = np.logical_and(g_inequality, np.logical_not(g_bounded))

        self.where_g_upper_bounded = np.where(g_upper_bounded)[0]
        self.where_g_lower_bounded = np.where(g_lower_bounded)[0]
        self.where_g_upper_lower_bounded = np.where(g_upper_lower_bounded)[0]
        self.where_g_bounded = np.where(g_bounded)[0]
        self.where_g_unbounded = np.where(g_unbounded)[0]                                   



        # Create symbolic variables for the Lagrange multipliers
        self.nlp["lag_x_all"] = lag_x_all = cd.SX.sym("lag_x_all", self.n_x, 1)
        self.nlp["lag_g_all"] = lag_g_all = cd.SX.sym("lag_g_all", self.n_g, 1)

        # Equality constraints
        self.nlp["eta_x"] = lag_x_all[self.where_x_equality]
        self.nlp["eta_g"] = lag_g_all[self.where_g_equality]
        self.nlp["eta"] = cd.vertcat(self.nlp["eta_g"], self.nlp["eta_x"])

        # Inequality constraints
        self.nlp["lam_x_upper_bounded"] = lag_x_all[self.where_x_upper_bounded]
        self.nlp["lam_x_lower_bounded"] = lag_x_all[self.where_x_lower_bounded]
        self.nlp["lam_x_upper_lower_bounded"] = lag_x_all[self.where_x_upper_lower_bounded]
        self.nlp["lam_x_bounded"] = lag_x_all[self.where_x_bounded]

        self.nlp["lam_g_upper_bounded"] = lag_g_all[self.where_g_upper_bounded]
        self.nlp["lam_g_lower_bounded"] = lag_g_all[self.where_g_lower_bounded]
        self.nlp["lam_g_upper_lower_bounded"] = lag_g_all[self.where_g_upper_lower_bounded]
        self.nlp["lam_g_bounded"] = lag_g_all[self.where_g_bounded]

        self.nlp["lam_bounded"] = cd.vertcat(self.nlp["lam_g_bounded"], self.nlp["lam_x_bounded"])

        self.eta_x_extractor = cd.Function("eta_x_extractor", [lag_x_all], [self.nlp["eta_x"]], ["lag_x_all"], ["eta_x"])
        self.eta_g_extractor = cd.Function("eta_g_extractor", [lag_g_all], [self.nlp["eta_g"]], ["lag_g_all"], ["eta_g"])
        self.lam_x_bounded_extractor = cd.Function("lam_x_bounded_extractor", [lag_x_all], [self.nlp["lam_x_bounded"]], ["lag_x_all"], ["lam_x_bounded"])
        self.lam_g_bounded_extractor = cd.Function("lam_g_bounded_extractor", [lag_g_all], [self.nlp["lam_g_bounded"]], ["lag_g_all"], ["lam_g_bounded"])


        self.nlp["z"] = cd.vertcat(self.nlp["x"], self.nlp["eta"], self.nlp["lam_bounded"])
        self.nlp["z_full"] = cd.vertcat(self.nlp["x"], self.nlp["lag_x_all"], self.nlp["lag_g_all"])
        self.n_z = self.nlp["z"].shape[0]
        self.n_z_full = self.nlp["z_full"].shape[0]
        self._opt_x_extractor = cd.Function("opt_x_extractor", [self.nlp["z"]], [self.nlp["x"]], ["z_opt"], ["opt_x"])


        self.L_sym = 0
        self.L_sym += self.nlp["f"]


        # Extract bounds
        lbx = self.nlp_bounds["lbx"]
        ubx = self.nlp_bounds["ubx"]

        lbg = self.nlp_bounds["lbg"]
        ubg = self.nlp_bounds["ubg"]


        # Equality constraints
        self.g_equality = g_equality = (self.nlp["g"][self.where_g_equality] - ubg[self.where_g_equality])
        self.x_equality = x_equality = (self.nlp["x"][self.where_x_equality] - ubx[self.where_x_equality])
        self.equality = cd.vertcat(g_equality, x_equality)
        self.n_h = self.equality.shape[0]

        self.L_sym += self.nlp["eta_g"].T @ g_equality
        self.L_sym += self.nlp["eta_x"].T @ x_equality


        #  Inequality constraints
        # Upper and lower bounded
        g_inequality = self.nlp["g"][self.where_g_upper_lower_bounded]
        g_inequality_upper = g_inequality - ubg[self.where_g_upper_lower_bounded]
        g_inequality_lower = lbg[self.where_g_upper_lower_bounded] - g_inequality
        g_inequality = cd.if_else(cd.fabs(g_inequality_upper) < cd.fabs(g_inequality_lower), g_inequality_upper, g_inequality_lower)
        self.g_inequality = g_inequality
        self.L_sym += cd.fabs(self.nlp["lam_g_upper_lower_bounded"]).T @ g_inequality

        # Upper bounded (This is proven to be correct)
        g_inequality = self.nlp["g"][self.where_g_upper_bounded]
        g_inequality = g_inequality - ubg[self.where_g_upper_bounded]
        self.g_inequality = cd.vertcat(self.g_inequality, g_inequality)
        self.L_sym += cd.fabs(self.nlp["lam_g_upper_bounded"]).T @ g_inequality

        # Lower bounded
        g_inequality = self.nlp["g"][self.where_g_lower_bounded]
        g_inequality = lbg[self.where_g_lower_bounded] - g_inequality
        self.g_inequality = cd.vertcat(self.g_inequality, g_inequality)
        self.L_sym += cd.fabs(self.nlp["lam_g_lower_bounded"]).T @ g_inequality
        self.n_g_ieq = self.g_inequality.shape[0]



        # Upper and lower bounded
        x_inequality = self.nlp["x"][self.where_x_upper_lower_bounded]
        x_inequality_upper = x_inequality - ubx[self.where_x_upper_lower_bounded]
        x_inequality_lower = lbx[self.where_x_upper_lower_bounded] - x_inequality
        x_inequality = cd.if_else(cd.fabs(x_inequality_upper) < cd.fabs(x_inequality_lower), x_inequality_upper, x_inequality_lower)
        self.x_inequality = x_inequality
        self.L_sym += cd.fabs(self.nlp["lam_x_upper_lower_bounded"]).T @ x_inequality

        # Upper bounded (According to g-term this is correct)
        x_inequality = self.nlp["x"][self.where_x_upper_bounded]
        x_inequality = x_inequality - ubx[self.where_x_upper_bounded]
        self.x_inequality = cd.vertcat(self.x_inequality, x_inequality)
        self.L_sym += cd.fabs(self.nlp["lam_x_upper_bounded"]).T @ x_inequality

        # Lower bounded
        x_inequality = self.nlp["x"][self.where_x_lower_bounded]
        x_inequality = lbx[self.where_x_lower_bounded] - x_inequality
        self.x_inequality = cd.vertcat(self.x_inequality, x_inequality)
        self.L_sym += cd.fabs(self.nlp["lam_x_lower_bounded"]).T @ x_inequality
        self.n_x_ieq = self.x_inequality.shape[0]

        self.inequality = cd.vertcat(self.g_inequality, self.x_inequality)
        self.inequality_func = cd.Function("inequality_func", [self.nlp["z_full"], self.nlp["p"]], [self.g_inequality, self.x_inequality], ["z_opt", "p"], ["g_inequality", "x_inequality"])

    def _prepare_sensitivity_matrices(self):
        """
        Warning:
            Not part of the public API.

        Calculates the sensitivity matrices of the NLP.
        """
        self.F = cd.gradient(self.L_sym, self.nlp["x"])
        self.F = cd.vertcat(self.F, self.equality)

        lam_g_inequality = cd.vertcat(self.nlp["lam_g_upper_lower_bounded"], self.nlp["lam_g_upper_bounded"], self.nlp["lam_g_lower_bounded"])
        self.F = cd.vertcat(self.F, cd.fabs(lam_g_inequality) * self.g_inequality) # NOTE: Here

        lam_x_inequality = cd.vertcat(self.nlp["lam_x_upper_lower_bounded"], self.nlp["lam_x_upper_bounded"], self.nlp["lam_x_lower_bounded"])
        self.F = cd.vertcat(self.F, cd.fabs(lam_x_inequality) * self.x_inequality) # NOTE: Here

        p = self.nlp["p"]
        RL_p = self._p_extractor(p)
        z = self.nlp["z"]
        z_full = self.nlp["z_full"]

        x = self.nlp["x"]
        lag_x_all = self.nlp["lag_x_all"]
        lag_g_all = self.nlp["lag_g_all"]


        # Setup A matrix
        self.A_sym = cd.jacobian(self.F, z)
        self.A_func = cd.Function("A", [x, lag_x_all, lag_g_all, p], [self.A_sym], ["x_opt", "lag_x_opt", "lag_g_opt", "p_opt"], ["A"])
        


        # Setup B matrix
        self.B_sym = cd.jacobian(self.F, RL_p)
        self.B_func = cd.Function("B", [x, lag_x_all, lag_g_all, p], [self.B_sym], ["x_opt", "lag_x_opt", "lag_g_opt", "p_opt"], ["B"])



        if self.settings.second_order:
            # NOTE: General remark on the computation of the second order sensitivities:
            # Since the second order sensitivities result in a symmetric submatrix (last 2 dims), the number of right hand side vectors in the C matrix can be reduced.
            # This is done by computing the sensitivities for each parameter separately and concatenating the results afterwards.
            # The number reduces from n_rhs = n_p ** 2 to n_rhs = n_p * (n_p + 1) / 2, which is roughly half the number of computations.
            # Further, the RAM consumption is roughly reduced by a factor of 2, which could result in a significant speedup for large problems due to excessive saving and loading times in multiprocessing.   
            # NOTE: This is not implemented yet, but should be done in the future.
            
            # Setup C matrix
            first_order_sensitivities = cd.SX.sym("first_order_sensitivities", (self.n_z, self.n_p))
            C = []

            # Refer to ECC24 paper for the derivation of the second order sensitivity matrices
            for idx_p in range(self.n_p):

                # E-part (equation (18d) from ECC24 paper)
                E_2 = cd.reshape(cd.jacobian(self.A_sym, RL_p[idx_p]), self.A_sym.shape)

                sens_row = first_order_sensitivities[:, idx_p]
                E_1 = cd.jacobian(self.A_sym, z) @ sens_row
                E_1 = cd.reshape(E_1, self.A_sym.shape)

                E = E_1 + E_2
                
                # D-part (equation (18c) from ECC24 paper)
                D_3 = cd.reshape(cd.jacobian(self.B_sym, RL_p[idx_p]), self.B_sym.shape)
                
                D_1 = cd.jacobian(self.B_sym, z) @ sens_row
                D_1 = cd.reshape(D_1, self.B_sym.shape)

                D = D_1 + D_3

                # C-part (equation (18b) from ECC24 paper)
                C.append(D + E @ first_order_sensitivities)
            
            # Concatenate the matrices (18a) from ECC24 paper
            C = cd.horzcat(*C)
            self.C_sym = cd.simplify(C)
            self.C_func = cd.Function("C_func", [z_full, p, first_order_sensitivities], [self.C_sym], ["z_opt", "p_opt", "first_order_sensitivities"], ["C"])

        self.status.sym_KKT = True

    def _detect_undetermined_sym_var(self, var: str = "x"): 
        
        # symbolic expressions
        var_sym = self.nlp[var]        
        # objective function
        f_sym = self.nlp["f"]
        # constraints
        g_sym = self.nlp["g"]

        # boolean expressions on wether a symbolic is contained in the objective function f or the constraints g
        map_f_var = map(lambda x: cd.depends_on(f_sym,x), cd.vertsplit(var_sym))
        map_g_var = map(lambda x: cd.depends_on(g_sym,x), cd.vertsplit(var_sym))

        # combined boolean expressions as list for each symbolic variable in var_sym
        dep_list = [f_dep or g_dep for f_dep,g_dep in zip(map_f_var,map_g_var)]

        # indices of undetermined and determined symbolic variables
        undet_sym_idx = np.where(np.logical_not(dep_list))[0]
        det_sym_idx = np.where(dep_list)[0]

        return undet_sym_idx, det_sym_idx
    
    def _get_do_mpc_nlp(self):
        """
        Warning:
            Not part of the public API.

        This function is used to extract the symbolic expressions and bounds of the underlying NLP of the MPC.
        It is used to initialize the NLPDifferentiator class.
        """

        # 1 get symbolic expressions of NLP
        nlp = {'x': cd.vertcat(self.mpc.opt_x), 'f': self.mpc.nlp_obj, 'g': self.mpc.nlp_cons, 'p': cd.vertcat(self.mpc.opt_p)}

        # 2 extract bounds
        nlp_bounds = {}
        nlp_bounds['lbg'] = self.mpc.nlp_cons_lb
        nlp_bounds['ubg'] = self.mpc.nlp_cons_ub
        nlp_bounds['lbx'] = cd.vertcat(self.mpc._lb_opt_x)
        nlp_bounds['ubx'] = cd.vertcat(self.mpc._ub_opt_x)

        return nlp, nlp_bounds

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes.pop("mpc")
        attributes.pop("nlp")
        attributes.pop("nlp_unreduced")
        attributes.pop("L_sym")
        attributes.pop("g_equality")
        attributes.pop("x_equality")
        attributes.pop("equality")
        attributes.pop("g_inequality")
        attributes.pop("x_inequality")
        attributes.pop("inequality")
        attributes.pop("F")
        attributes.pop("A_sym")
        attributes.pop("B_sym")
        if self.settings.second_order:
            attributes.pop("C_sym")
        return attributes
    
    def __setstate__(self, state):
        self.__dict__.update(state)