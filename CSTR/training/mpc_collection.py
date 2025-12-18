# Import necessary modules
from do_mpc.model import Model
from RL_MPC import RL_MPC
import casadi as cd

def get_rl_mpc(penalty_weight = 1e2):
    # Define model type
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = Model(model_type)

    # Define state variables (optimization variables)
    C_a = model.set_variable(var_type='_x', var_name='C_a', shape=(1,1))
    C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1,1))
    T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1,1))
    T_K = model.set_variable(var_type='_x', var_name='T_K', shape=(1,1))

    # Define input variables (optimization variables)
    F = model.set_variable(var_type='_u', var_name='F')
    Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')

    # Define uncertain parameters (\theta_\alpha, \theta_\beta in the paper)
    alpha = model.set_variable(var_type = "_p", var_name= "alpha", shape = (1,1)) # Uncertain parameter
    beta = model.set_variable(var_type = "_p", var_name= "beta", shape = (1,1)) # Uncertain parameter

    # Parameterization (\tilde{\mu}_\alpha}, \tilde{\mu}_\beta in the paper)
    alpha = (1 + alpha) * 0.95
    beta = (1 + beta) * 1.0

    # Define certain parameters
    K0_ab = 1.287e12 # K0 [h^-1]
    K0_bc = 1.287e12 # K0 [h^-1]
    K0_ad = 9.043e9 # K0 [l/mol.h]
    R_gas = 8.3144621e-3 # Universal gas constant
    E_A_ab = 9758.3*1.00 # [kj/mol]
    E_A_bc = 9758.3*1.00 # [kj/mol]
    E_A_ad = 8560.0*1.0 # [kj/mol]
    H_R_ab = 4.2 # [kj/mol A]
    H_R_bc = -11.0 # [kj/mol B] Exothermic
    H_R_ad = -41.85 # [kj/mol A] Exothermic
    Rou = 0.9342 # Density [kg/l]
    Cp = 3.01 # Specific Heat capacity [kj/Kg.K]
    Cp_k = 2.0 # Coolant heat capacity [kj/kg.k]
    A_R = 0.215 # Area of reactor wall [m^2]
    V_R = 10.01 # Volume of reactor [l]
    m_k = 5.0 # Coolant mass[kg]
    T_in = 130.0 # Temp of inflow [Celsius]
    K_w = 4032.0 # [kj/h.m^2.K]
    C_A0 = (5.7+4.5)/2.0*1.0 # Concentration of A in input [mol/l]

    # Auxiliary terms for reaction rates
    K_1 = beta * K0_ab * cd.exp((-E_A_ab)/((T_R+273.15)))
    K_2 =  K0_bc * cd.exp((-E_A_bc)/((T_R+273.15)))
    K_3 = K0_ad * cd.exp((-alpha*E_A_ad)/((T_R+273.15)))

    # Define temperature difference expression
    T_dif = model.set_expression(expr_name='T_dif', expr=T_R-T_K)

    # Set right-hand side of ODEs for states
    model.set_rhs('C_a', F*(C_A0 - C_a) -K_1*C_a - K_3*(C_a**2))
    model.set_rhs('C_b', -F*C_b + K_1*C_a - K_2*C_b)
    model.set_rhs('T_R', ((K_1*C_a*H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2)*H_R_ad)/(-Rou*Cp)) + F*(T_in-T_R) +(((K_w*A_R)*(-T_dif))/(Rou*Cp*V_R)))
    model.set_rhs('T_K', (Q_dot + K_w*A_R*(T_dif))/(m_k*Cp_k))
    
    # Finalize model setup
    model.setup()

    # Create RL_MPC controller
    mpc = RL_MPC(model)

    # MPC settings
    mpc.settings.n_horizon = 20
    mpc.settings.open_loop = 0
    mpc.settings.t_step = 0.005
    mpc.settings.state_discretization = "collocation"
    mpc.settings.collocation_type = "radau"
    mpc.settings.collocation_deg = 2
    mpc.settings.collocation_ni = 2
    mpc.settings.nlpsol_opts["ipopt.print_level"] = 0
    mpc.settings.nlpsol_opts["print_time"] = 0
    mpc.settings.nlpsol_opts['ipopt.sb'] = 'yes'

    # Scaling for states and inputs
    mpc.scaling['_x', 'T_R'] = 100
    mpc.scaling['_x', 'T_K'] = 100
    mpc.scaling['_u', 'Q_dot'] = 2000
    mpc.scaling['_u', 'F'] = 100

    # Objective function (stage and terminal cost)
    lterm = 1e-2 * (T_R - 126) ** 2 + 1e-2 * (T_K - 120.0) ** 2
    mterm = 1e-2 * (T_R - 126) ** 2 + 1e-2 * (T_K - 120.0) ** 2
    mpc.set_objective(mterm = mterm, lterm = lterm)

    # Nonlinear constraints (soft constraints with penalties to always ensure feasibility)
    mpc.set_nl_cons(expr_name = "C_a_lower", expr = 0.1 - C_a, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "C_a_upper", expr = C_a - 2.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "C_b_lower", expr = 0.1 - C_b, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "C_b_upper", expr = C_b - 2.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_R_lower", expr = 80 - T_R, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_R_upper", expr = T_R - 140.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_K_lower", expr = 80 - T_K, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_K_upper", expr = T_K - 140.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    
    # Input bounds
    mpc.bounds['lower', '_u', 'F'] = 5
    mpc.bounds['lower', '_u', 'Q_dot'] = -8500
    mpc.bounds['upper', '_u', 'F'] = 40
    mpc.bounds['upper', '_u', 'Q_dot'] = 0.0

    # Flag for regularization term
    mpc.flags["set_rterm"] = True

    # Parameter template for uncertain parameters
    p_template = mpc.get_p_template(1)
    mpc.set_p_fun(lambda t_now: p_template)

    # Prepare NLP problem
    mpc.prepare_nlp()

    # Add regularization term for control input changes
    u_prev = mpc.opt_p["_u_prev"]
    R = 1e2 * cd.diag([1e0, 1e0])  # Regularization matrix for the control input
    for idx in range(mpc.settings.n_horizon):
        u_now = mpc.opt_x_unscaled["_u", idx, 0]
        delta_u = u_now - u_prev
        delta_u = delta_u / (mpc._u_ub.master - mpc._u_lb.master)
        r_term = delta_u.T @ R @ delta_u  # Regularization term for the control input
        u_prev = u_now
        mpc._nlp_obj += r_term

    # Finalize NLP creation
    mpc.create_nlp()

    return mpc

def get_rl_mpc_scaled_params(penalty_weight = 1e2):

    # Define model type
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = Model(model_type)

    # Define state variables (optimization variables)
    C_a = model.set_variable(var_type='_x', var_name='C_a', shape=(1,1))
    C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1,1))
    T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1,1))
    T_K = model.set_variable(var_type='_x', var_name='T_K', shape=(1,1))

    # Define input variables (optimization variables)
    F = model.set_variable(var_type='_u', var_name='F')
    Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')

    # Define uncertain parameters (\theta_\alpha, \theta_\beta in the paper)
    alpha = model.set_variable(var_type = "_p", var_name= "alpha", shape = (1,1)) # Uncertain parameter
    beta = model.set_variable(var_type = "_p", var_name= "beta", shape = (1,1)) # Uncertain parameter

    # Parameterization (\tilde{\mu}_\alpha}, \tilde{\mu}_\beta in the paper)
    alpha = (1 + 0.1 * alpha) * 0.95
    beta = (1 + beta) * 1.0

    # Define certain parameters
    K0_ab = 1.287e12 # K0 [h^-1]
    K0_bc = 1.287e12 # K0 [h^-1]
    K0_ad = 9.043e9 # K0 [l/mol.h]
    R_gas = 8.3144621e-3 # Universal gas constant
    E_A_ab = 9758.3*1.00 # [kj/mol]
    E_A_bc = 9758.3*1.00 # [kj/mol]
    E_A_ad = 8560.0*1.0 # [kj/mol]
    H_R_ab = 4.2 # [kj/mol A]
    H_R_bc = -11.0 # [kj/mol B] Exothermic
    H_R_ad = -41.85 # [kj/mol A] Exothermic
    Rou = 0.9342 # Density [kg/l]
    Cp = 3.01 # Specific Heat capacity [kj/Kg.K]
    Cp_k = 2.0 # Coolant heat capacity [kj/kg.k]
    A_R = 0.215 # Area of reactor wall [m^2]
    V_R = 10.01 # Volume of reactor [l]
    m_k = 5.0 # Coolant mass[kg]
    T_in = 130.0 # Temp of inflow [Celsius]
    K_w = 4032.0 # [kj/h.m^2.K]
    C_A0 = (5.7+4.5)/2.0*1.0 # Concentration of A in input [mol/l]

    # Auxiliary terms for reaction rates
    K_1 = beta * K0_ab * cd.exp((-E_A_ab)/((T_R+273.15)))
    K_2 =  K0_bc * cd.exp((-E_A_bc)/((T_R+273.15)))
    K_3 = K0_ad * cd.exp((-alpha*E_A_ad)/((T_R+273.15)))

    # Define temperature difference expression
    T_dif = model.set_expression(expr_name='T_dif', expr=T_R-T_K)

    # Set right-hand side of ODEs for states
    model.set_rhs('C_a', F*(C_A0 - C_a) -K_1*C_a - K_3*(C_a**2))
    model.set_rhs('C_b', -F*C_b + K_1*C_a - K_2*C_b)
    model.set_rhs('T_R', ((K_1*C_a*H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2)*H_R_ad)/(-Rou*Cp)) + F*(T_in-T_R) +(((K_w*A_R)*(-T_dif))/(Rou*Cp*V_R)))
    model.set_rhs('T_K', (Q_dot + K_w*A_R*(T_dif))/(m_k*Cp_k))
    
    # Finalize model setup
    model.setup()

    # Create RL_MPC controller
    mpc = RL_MPC(model)

    # MPC settings
    mpc.settings.n_horizon = 20
    mpc.settings.open_loop = 0
    mpc.settings.t_step = 0.005
    mpc.settings.state_discretization = "collocation"
    mpc.settings.collocation_type = "radau"
    mpc.settings.collocation_deg = 2
    mpc.settings.collocation_ni = 2
    mpc.settings.nlpsol_opts["ipopt.print_level"] = 0
    mpc.settings.nlpsol_opts["print_time"] = 0
    mpc.settings.nlpsol_opts['ipopt.sb'] = 'yes'

    # Scaling for states and inputs
    mpc.scaling['_x', 'T_R'] = 100
    mpc.scaling['_x', 'T_K'] = 100
    mpc.scaling['_u', 'Q_dot'] = 2000
    mpc.scaling['_u', 'F'] = 100

    # Objective function (stage and terminal cost)
    lterm = 1e-2 * (T_R - 126) ** 2 + 1e-2 * (T_K - 120.0) ** 2
    mterm = 1e-2 * (T_R - 126) ** 2 + 1e-2 * (T_K - 120.0) ** 2
    mpc.set_objective(mterm = mterm, lterm = lterm)

    # Nonlinear constraints (soft constraints with penalties to always ensure feasibility)
    mpc.set_nl_cons(expr_name = "C_a_lower", expr = 0.1 - C_a, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "C_a_upper", expr = C_a - 2.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "C_b_lower", expr = 0.1 - C_b, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "C_b_upper", expr = C_b - 2.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_R_lower", expr = 80 - T_R, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_R_upper", expr = T_R - 140.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_K_lower", expr = 80 - T_K, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_K_upper", expr = T_K - 140.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    
    # Input bounds
    mpc.bounds['lower', '_u', 'F'] = 5
    mpc.bounds['lower', '_u', 'Q_dot'] = -8500
    mpc.bounds['upper', '_u', 'F'] = 40
    mpc.bounds['upper', '_u', 'Q_dot'] = 0.0

    # Flag for regularization term
    mpc.flags["set_rterm"] = True

    # Parameter template for uncertain parameters
    p_template = mpc.get_p_template(1)
    mpc.set_p_fun(lambda t_now: p_template)

    # Prepare NLP problem
    mpc.prepare_nlp()

    # Add regularization term for control input changes
    u_prev = mpc.opt_p["_u_prev"]
    R = 1e2 * cd.diag([1e0, 1e0])  # Regularization matrix for the control input
    for idx in range(mpc.settings.n_horizon):
        u_now = mpc.opt_x_unscaled["_u", idx, 0]
        delta_u = u_now - u_prev
        delta_u = delta_u / (mpc._u_ub.master - mpc._u_lb.master)
        r_term = delta_u.T @ R @ delta_u  # Regularization term for the control input
        u_prev = u_now
        mpc._nlp_obj += r_term

    # Finalize NLP creation
    mpc.create_nlp()

    return mpc

def get_rl_mpc_malscaled_params(penalty_weight = 1e2):

    # Define model type
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = Model(model_type)

    # Define state variables (optimization variables)
    C_a = model.set_variable(var_type='_x', var_name='C_a', shape=(1,1))
    C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1,1))
    T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1,1))
    T_K = model.set_variable(var_type='_x', var_name='T_K', shape=(1,1))

    # Define input variables (optimization variables)
    F = model.set_variable(var_type='_u', var_name='F')
    Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')

    # Define uncertain parameters (\theta_\alpha, \theta_\beta in the paper)
    alpha = model.set_variable(var_type = "_p", var_name= "alpha", shape = (1,1)) # Uncertain parameter
    beta = model.set_variable(var_type = "_p", var_name= "beta", shape = (1,1)) # Uncertain parameter

    # Parameterization (\tilde{\mu}_\alpha}, \tilde{\mu}_\beta in the paper)
    alpha = (1 + 10.0 * alpha) * 0.95
    beta = (1 + beta) * 1.0

    # Define certain parameters
    K0_ab = 1.287e12 # K0 [h^-1]
    K0_bc = 1.287e12 # K0 [h^-1]
    K0_ad = 9.043e9 # K0 [l/mol.h]
    R_gas = 8.3144621e-3 # Universal gas constant
    E_A_ab = 9758.3*1.00 # [kj/mol]
    E_A_bc = 9758.3*1.00 # [kj/mol]
    E_A_ad = 8560.0*1.0 # [kj/mol]
    H_R_ab = 4.2 # [kj/mol A]
    H_R_bc = -11.0 # [kj/mol B] Exothermic
    H_R_ad = -41.85 # [kj/mol A] Exothermic
    Rou = 0.9342 # Density [kg/l]
    Cp = 3.01 # Specific Heat capacity [kj/Kg.K]
    Cp_k = 2.0 # Coolant heat capacity [kj/kg.k]
    A_R = 0.215 # Area of reactor wall [m^2]
    V_R = 10.01 # Volume of reactor [l]
    m_k = 5.0 # Coolant mass[kg]
    T_in = 130.0 # Temp of inflow [Celsius]
    K_w = 4032.0 # [kj/h.m^2.K]
    C_A0 = (5.7+4.5)/2.0*1.0 # Concentration of A in input [mol/l]

    # Auxiliary terms for reaction rates
    K_1 = beta * K0_ab * cd.exp((-E_A_ab)/((T_R+273.15)))
    K_2 =  K0_bc * cd.exp((-E_A_bc)/((T_R+273.15)))
    K_3 = K0_ad * cd.exp((-alpha*E_A_ad)/((T_R+273.15)))

    # Define temperature difference expression
    T_dif = model.set_expression(expr_name='T_dif', expr=T_R-T_K)

    # Set right-hand side of ODEs for states
    model.set_rhs('C_a', F*(C_A0 - C_a) -K_1*C_a - K_3*(C_a**2))
    model.set_rhs('C_b', -F*C_b + K_1*C_a - K_2*C_b)
    model.set_rhs('T_R', ((K_1*C_a*H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2)*H_R_ad)/(-Rou*Cp)) + F*(T_in-T_R) +(((K_w*A_R)*(-T_dif))/(Rou*Cp*V_R)))
    model.set_rhs('T_K', (Q_dot + K_w*A_R*(T_dif))/(m_k*Cp_k))
    
    # Finalize model setup
    model.setup()

    # Create RL_MPC controller
    mpc = RL_MPC(model)

    # MPC settings
    mpc.settings.n_horizon = 20
    mpc.settings.open_loop = 0
    mpc.settings.t_step = 0.005
    mpc.settings.state_discretization = "collocation"
    mpc.settings.collocation_type = "radau"
    mpc.settings.collocation_deg = 2
    mpc.settings.collocation_ni = 2
    mpc.settings.nlpsol_opts["ipopt.print_level"] = 0
    mpc.settings.nlpsol_opts["print_time"] = 0
    mpc.settings.nlpsol_opts['ipopt.sb'] = 'yes'

    # Scaling for states and inputs
    mpc.scaling['_x', 'T_R'] = 100
    mpc.scaling['_x', 'T_K'] = 100
    mpc.scaling['_u', 'Q_dot'] = 2000
    mpc.scaling['_u', 'F'] = 100

    # Objective function (stage and terminal cost)
    lterm = 1e-2 * (T_R - 126) ** 2 + 1e-2 * (T_K - 120.0) ** 2
    mterm = 1e-2 * (T_R - 126) ** 2 + 1e-2 * (T_K - 120.0) ** 2
    mpc.set_objective(mterm = mterm, lterm = lterm)

    # Nonlinear constraints (soft constraints with penalties to always ensure feasibility)
    mpc.set_nl_cons(expr_name = "C_a_lower", expr = 0.1 - C_a, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "C_a_upper", expr = C_a - 2.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "C_b_lower", expr = 0.1 - C_b, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "C_b_upper", expr = C_b - 2.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_R_lower", expr = 80 - T_R, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_R_upper", expr = T_R - 140.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_K_lower", expr = 80 - T_K, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_K_upper", expr = T_K - 140.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    
    # Input bounds
    mpc.bounds['lower', '_u', 'F'] = 5
    mpc.bounds['lower', '_u', 'Q_dot'] = -8500
    mpc.bounds['upper', '_u', 'F'] = 40
    mpc.bounds['upper', '_u', 'Q_dot'] = 0.0

    # Flag for regularization term
    mpc.flags["set_rterm"] = True

    # Parameter template for uncertain parameters
    p_template = mpc.get_p_template(1)
    mpc.set_p_fun(lambda t_now: p_template)

    # Prepare NLP problem
    mpc.prepare_nlp()

    # Add regularization term for control input changes
    u_prev = mpc.opt_p["_u_prev"]
    R = 1e2 * cd.diag([1e0, 1e0])  # Regularization matrix for the control input
    for idx in range(mpc.settings.n_horizon):
        u_now = mpc.opt_x_unscaled["_u", idx, 0]
        delta_u = u_now - u_prev
        delta_u = delta_u / (mpc._u_ub.master - mpc._u_lb.master)
        r_term = delta_u.T @ R @ delta_u  # Regularization term for the control input
        u_prev = u_now
        mpc._nlp_obj += r_term

    # Finalize NLP creation
    mpc.create_nlp()

    return mpc

def get_rl_mpc_medium_parameterized(penalty_weight = 1e2):
    # Define model type
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = Model(model_type)

    # Define state variables (optimization variables)
    C_a = model.set_variable(var_type='_x', var_name='C_a', shape=(1,1))
    C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1,1))
    T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1,1))
    T_K = model.set_variable(var_type='_x', var_name='T_K', shape=(1,1))

    # Define input variables (optimization variables)
    F = model.set_variable(var_type='_u', var_name='F')
    Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')

    # Define uncertain parameters that should be optimized
    alpha = model.set_variable(var_type = "_p", var_name= "alpha", shape = (1,1)) # Uncertain parameter
    beta = model.set_variable(var_type = "_p", var_name= "beta", shape = (1,1)) # Uncertain parameter
    k_T_R_ref = model.set_variable(var_type = "_p", var_name= "k_T_R_ref", shape = (1,1)) # Uncertain parameter
    k_T_K_ref = model.set_variable(var_type = "_p", var_name= "k_T_K_ref", shape = (1,1)) # Uncertain parameter
    theta_K0_bc = model.set_variable(var_type = "_p", var_name= "theta_K0_bc", shape = (1,1)) # Uncertain parameter
    theta_K0_ad = model.set_variable(var_type = "_p", var_name= "theta_K0_ad", shape = (1,1)) # Uncertain parameter
    theta_E_A_ab = model.set_variable(var_type = "_p", var_name= "theta_E_A_ab", shape = (1,1)) # Uncertain parameter
    theta_E_A_bc = model.set_variable(var_type = "_p", var_name= "theta_E_A_bc", shape = (1,1)) # Uncertain parameter
    theta_H_R_ab = model.set_variable(var_type = "_p", var_name= "theta_H_R_ab", shape = (1,1)) # Uncertain parameter
    theta_H_R_bc = model.set_variable(var_type = "_p", var_name= "theta_H_R_bc", shape = (1,1)) # Uncertain parameter
    theta_H_R_ad = model.set_variable(var_type = "_p", var_name= "theta_H_R_ad", shape = (1,1)) # Uncertain parameter
    theta_Cp = model.set_variable(var_type = "_p", var_name= "theta_Cp", shape = (1,1)) # Uncertain parameter
    theta_Cp_k = model.set_variable(var_type = "_p", var_name= "theta_Cp_k", shape = (1,1)) # Uncertain parameter

    # Parameterization
    alpha = (1 + alpha) * 0.95
    beta = (1 + beta) * 1.0
    k_T_R_ref = (1 + k_T_R_ref) * 1.0
    k_T_K_ref = (1 + k_T_K_ref) * 1.0
    theta_K0_bc = (1 + theta_K0_bc) * 1.0
    theta_K0_ad = (1 + theta_K0_ad) * 1.0
    theta_E_A_ab = (1 + theta_E_A_ab) * 0.95
    theta_E_A_bc = (1 + theta_E_A_bc) * 1.05
    theta_H_R_ab = (1 + theta_H_R_ab) * 1.05
    theta_H_R_bc = (1 + theta_H_R_bc) * 1.05
    theta_H_R_ad = (1 + theta_H_R_ad) * 0.95
    theta_Cp = (1 + theta_Cp) * 0.98
    theta_Cp_k = (1 + theta_Cp_k) * 1.02

    # Define certain parameters
    K0_ab = 1.287e12 # K0 [h^-1]
    K0_bc = theta_K0_bc * 1.287e12 # K0 [h^-1]
    K0_ad = theta_K0_ad * 9.043e9 # K0 [l/mol.h]
    R_gas = 8.3144621e-3 # Universal gas constant
    E_A_ab = theta_E_A_ab * 9758.3*1.00 # [kj/mol]
    E_A_bc = theta_E_A_bc * 9758.3*1.00 # [kj/mol]
    E_A_ad = 8560.0*1.0 # [kj/mol]
    H_R_ab = theta_H_R_ab * 4.2 # [kj/mol A]
    H_R_bc = theta_H_R_bc * -11.0 # [kj/mol B] Exothermic
    H_R_ad = theta_H_R_ad * -41.85 # [kj/mol A] Exothermic
    Rou = 0.9342 # Density [kg/l]
    Cp = theta_Cp * 3.01 # Specific Heat capacity [kj/Kg.K]
    Cp_k = theta_Cp_k * 2.0 # Coolant heat capacity [kj/kg.k]
    A_R = 0.215 # Area of reactor wall [m^2]
    V_R = 10.01 # Volume of reactor [l]
    m_k = 5.0 # Coolant mass[kg]
    T_in = 130.0 # Temp of inflow [Celsius]
    K_w = 4032.0 # [kj/h.m^2.K]
    C_A0 = (5.7+4.5)/2.0*1.0 # Concentration of A in input [mol/l]

    # Auxiliary terms for reaction rates
    K_1 = beta * K0_ab * cd.exp((-E_A_ab)/((T_R+273.15)))
    K_2 =  K0_bc * cd.exp((-E_A_bc)/((T_R+273.15)))
    K_3 = K0_ad * cd.exp((-alpha*E_A_ad)/((T_R+273.15)))

    # Define temperature difference expression
    T_dif = model.set_expression(expr_name='T_dif', expr=T_R-T_K)

    # Set right-hand side of ODEs for states
    model.set_rhs('C_a', F*(C_A0 - C_a) -K_1*C_a - K_3*(C_a**2))
    model.set_rhs('C_b', -F*C_b + K_1*C_a - K_2*C_b)
    model.set_rhs('T_R', ((K_1*C_a*H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2)*H_R_ad)/(-Rou*Cp)) + F*(T_in-T_R) +(((K_w*A_R)*(-T_dif))/(Rou*Cp*V_R)))
    model.set_rhs('T_K', (Q_dot + K_w*A_R*(T_dif))/(m_k*Cp_k))
    
    # Finalize model setup
    model.setup()

    # Create RL_MPC controller
    mpc = RL_MPC(model)

    # MPC settings
    mpc.settings.n_horizon = 20
    mpc.settings.open_loop = 0
    mpc.settings.t_step = 0.005
    mpc.settings.state_discretization = "collocation"
    mpc.settings.collocation_type = "radau"
    mpc.settings.collocation_deg = 2
    mpc.settings.collocation_ni = 2
    mpc.settings.nlpsol_opts["ipopt.print_level"] = 0
    mpc.settings.nlpsol_opts["print_time"] = 0
    mpc.settings.nlpsol_opts['ipopt.sb'] = 'yes'

    # Scaling for states and inputs
    mpc.scaling['_x', 'T_R'] = 100
    mpc.scaling['_x', 'T_K'] = 100
    mpc.scaling['_u', 'Q_dot'] = 2000
    mpc.scaling['_u', 'F'] = 100

    k_T_K_ref = mpc.model._p['k_T_K_ref']
    k_T_R_ref = mpc.model._p['k_T_R_ref']
    k_T_R_ref = (1 + k_T_R_ref) * 1.0
    k_T_K_ref = (1 + k_T_K_ref) * 1.0

    # Objective function (stage and terminal cost)
    lterm = 1e-2 * (T_R - 126 * k_T_R_ref) ** 2 + 1e-2 * (T_K - 120.0 * k_T_K_ref) ** 2
    mterm = 1e-2 * (T_R - 126 * k_T_R_ref) ** 2 + 1e-2 * (T_K - 120.0 * k_T_K_ref) ** 2
    mpc.set_objective(mterm = mterm, lterm = lterm)

    # Nonlinear constraints (soft constraints with penalties to always ensure feasibility)
    mpc.set_nl_cons(expr_name = "C_a_lower", expr = 0.1 - C_a, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "C_a_upper", expr = C_a - 2.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "C_b_lower", expr = 0.1 - C_b, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "C_b_upper", expr = C_b - 2.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_R_lower", expr = 80 - T_R, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_R_upper", expr = T_R - 140.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_K_lower", expr = 80 - T_K, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_K_upper", expr = T_K - 140.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    
    # Input bounds
    mpc.bounds['lower', '_u', 'F'] = 5
    mpc.bounds['lower', '_u', 'Q_dot'] = -8500
    mpc.bounds['upper', '_u', 'F'] = 40
    mpc.bounds['upper', '_u', 'Q_dot'] = 0.0

    # Flag for regularization term
    mpc.flags["set_rterm"] = True

    # Parameter template for uncertain parameters
    p_template = mpc.get_p_template(1)
    mpc.set_p_fun(lambda t_now: p_template)

    # Prepare NLP problem
    mpc.prepare_nlp()

    # Add regularization term for control input changes
    u_prev = mpc.opt_p["_u_prev"]
    R = 1e2 * cd.diag([1e0, 1e0])  # Regularization matrix for the control input
    for idx in range(mpc.settings.n_horizon):
        u_now = mpc.opt_x_unscaled["_u", idx, 0]
        delta_u = u_now - u_prev
        delta_u = delta_u / (mpc._u_ub.master - mpc._u_lb.master)
        r_term = delta_u.T @ R @ delta_u  # Regularization term for the control input
        u_prev = u_now
        mpc._nlp_obj += r_term

    # Finalize NLP creation
    mpc.create_nlp()

    return mpc

def get_rl_mpc_high_parameterized(penalty_weight = 1e2):
    # Define model type
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = Model(model_type)

    # Define state variables (optimization variables)
    C_a = model.set_variable(var_type='_x', var_name='C_a', shape=(1,1))
    C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1,1))
    T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1,1))
    T_K = model.set_variable(var_type='_x', var_name='T_K', shape=(1,1))

    # Define input variables (optimization variables)
    F = model.set_variable(var_type='_u', var_name='F')
    Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')

    # Define uncertain parameters that should be optimized
    alpha = model.set_variable(var_type = "_p", var_name= "alpha", shape = (1,1)) # Uncertain parameter
    beta = model.set_variable(var_type = "_p", var_name= "beta", shape = (1,1)) # Uncertain parameter
    k_T_R_ref = model.set_variable(var_type = "_p", var_name= "k_T_R_ref", shape = (1,1)) # Uncertain parameter
    k_T_K_ref = model.set_variable(var_type = "_p", var_name= "k_T_K_ref", shape = (1,1)) # Uncertain parameter
    theta_K0_bc = model.set_variable(var_type = "_p", var_name= "theta_K0_bc", shape = (1,1)) # Uncertain parameter
    theta_K0_ad = model.set_variable(var_type = "_p", var_name= "theta_K0_ad", shape = (1,1)) # Uncertain parameter
    theta_E_A_ab = model.set_variable(var_type = "_p", var_name= "theta_E_A_ab", shape = (1,1)) # Uncertain parameter
    theta_E_A_bc = model.set_variable(var_type = "_p", var_name= "theta_E_A_bc", shape = (1,1)) # Uncertain parameter
    theta_H_R_ab = model.set_variable(var_type = "_p", var_name= "theta_H_R_ab", shape = (1,1)) # Uncertain parameter
    theta_H_R_bc = model.set_variable(var_type = "_p", var_name= "theta_H_R_bc", shape = (1,1)) # Uncertain parameter
    theta_H_R_ad = model.set_variable(var_type = "_p", var_name= "theta_H_R_ad", shape = (1,1)) # Uncertain parameter
    theta_Cp = model.set_variable(var_type = "_p", var_name= "theta_Cp", shape = (1,1)) # Uncertain parameter
    theta_Cp_k = model.set_variable(var_type = "_p", var_name= "theta_Cp_k", shape = (1,1)) # Uncertain parameter
    theta_P_term = model.set_variable(var_type = "_p", var_name= "theta_P_term", shape = (4,4)) # Uncertain parameter
    theta_x_term_set = model.set_variable(var_type = "_p", var_name= "theta_x_term_set", shape = (4,1)) # Uncertain parameter

    # Parameterization
    alpha = (1 + alpha) * 0.95
    beta = (1 + beta) * 1.0
    k_T_R_ref = (1 + k_T_R_ref) * 1.0
    k_T_K_ref = (1 + k_T_K_ref) * 1.0
    theta_K0_bc = (1 + theta_K0_bc) * 1.0
    theta_K0_ad = (1 + theta_K0_ad) * 1.0
    theta_E_A_ab = (1 + theta_E_A_ab) * 0.95
    theta_E_A_bc = (1 + theta_E_A_bc) * 1.05
    theta_H_R_ab = (1 + theta_H_R_ab) * 1.05
    theta_H_R_bc = (1 + theta_H_R_bc) * 1.05
    theta_H_R_ad = (1 + theta_H_R_ad) * 0.95
    theta_Cp = (1 + theta_Cp) * 0.98
    theta_Cp_k = (1 + theta_Cp_k) * 1.02

    # Define certain parameters
    K0_ab = 1.287e12 # K0 [h^-1]
    K0_bc = theta_K0_bc * 1.287e12 # K0 [h^-1]
    K0_ad = theta_K0_ad * 9.043e9 # K0 [l/mol.h]
    R_gas = 8.3144621e-3 # Universal gas constant
    E_A_ab = theta_E_A_ab * 9758.3*1.00 # [kj/mol]
    E_A_bc = theta_E_A_bc * 9758.3*1.00 # [kj/mol]
    E_A_ad = 8560.0*1.0 # [kj/mol]
    H_R_ab = theta_H_R_ab * 4.2 # [kj/mol A]
    H_R_bc = theta_H_R_bc * -11.0 # [kj/mol B] Exothermic
    H_R_ad = theta_H_R_ad * -41.85 # [kj/mol A] Exothermic
    Rou = 0.9342 # Density [kg/l]
    Cp = theta_Cp * 3.01 # Specific Heat capacity [kj/Kg.K]
    Cp_k = theta_Cp_k * 2.0 # Coolant heat capacity [kj/kg.k]
    A_R = 0.215 # Area of reactor wall [m^2]
    V_R = 10.01 # Volume of reactor [l]
    m_k = 5.0 # Coolant mass[kg]
    T_in = 130.0 # Temp of inflow [Celsius]
    K_w = 4032.0 # [kj/h.m^2.K]
    C_A0 = (5.7+4.5)/2.0*1.0 # Concentration of A in input [mol/l]

    # Auxiliary terms for reaction rates
    K_1 = beta * K0_ab * cd.exp((-E_A_ab)/((T_R+273.15)))
    K_2 =  K0_bc * cd.exp((-E_A_bc)/((T_R+273.15)))
    K_3 = K0_ad * cd.exp((-alpha*E_A_ad)/((T_R+273.15)))

    # Define temperature difference expression
    T_dif = model.set_expression(expr_name='T_dif', expr=T_R-T_K)

    # Set right-hand side of ODEs for states
    model.set_rhs('C_a', F*(C_A0 - C_a) -K_1*C_a - K_3*(C_a**2))
    model.set_rhs('C_b', -F*C_b + K_1*C_a - K_2*C_b)
    model.set_rhs('T_R', ((K_1*C_a*H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2)*H_R_ad)/(-Rou*Cp)) + F*(T_in-T_R) +(((K_w*A_R)*(-T_dif))/(Rou*Cp*V_R)))
    model.set_rhs('T_K', (Q_dot + K_w*A_R*(T_dif))/(m_k*Cp_k))
    
    # Finalize model setup
    model.setup()

    # Create RL_MPC controller
    mpc = RL_MPC(model)

    # MPC settings
    mpc.settings.n_horizon = 20
    mpc.settings.open_loop = 0
    mpc.settings.t_step = 0.005
    mpc.settings.state_discretization = "collocation"
    mpc.settings.collocation_type = "radau"
    mpc.settings.collocation_deg = 2
    mpc.settings.collocation_ni = 2
    mpc.settings.nlpsol_opts["ipopt.print_level"] = 0
    mpc.settings.nlpsol_opts["print_time"] = 0
    mpc.settings.nlpsol_opts['ipopt.sb'] = 'yes'

    # Scaling for states and inputs
    mpc.scaling['_x', 'T_R'] = 100
    mpc.scaling['_x', 'T_K'] = 100
    mpc.scaling['_u', 'Q_dot'] = 2000
    mpc.scaling['_u', 'F'] = 100

    k_T_K_ref = mpc.model._p['k_T_K_ref']
    k_T_R_ref = mpc.model._p['k_T_R_ref']
    k_T_R_ref = (1 + k_T_R_ref) * 1.0
    k_T_K_ref = (1 + k_T_K_ref) * 1.0

    theta_P_term = mpc.model._p['theta_P_term']
    theta_P_term = cd.diag([0, 0, 1e-2, 1e-2]) + theta_P_term
    theta_x_term_set = mpc.model._p['theta_x_term_set']
    theta_x_term_set = cd.DM(([0.5, 0.5, 126, 120.0])).reshape((4,1)) * (1 + theta_x_term_set)

    _x = mpc.model._x.master

    # Objective function (stage and terminal cost)
    lterm = 1e-2 * (T_R - 126 * k_T_R_ref) ** 2 + 1e-2 * (T_K - 120.0 * k_T_K_ref) ** 2
    mterm = (_x  - theta_x_term_set).T @ theta_P_term.T @ theta_P_term @ (_x - theta_x_term_set)
    mpc.set_objective(mterm = mterm, lterm = lterm)

    # Nonlinear constraints (soft constraints with penalties to always ensure feasibility)
    mpc.set_nl_cons(expr_name = "C_a_lower", expr = 0.1 - C_a, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "C_a_upper", expr = C_a - 2.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "C_b_lower", expr = 0.1 - C_b, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "C_b_upper", expr = C_b - 2.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_R_lower", expr = 80 - T_R, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_R_upper", expr = T_R - 140.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_K_lower", expr = 80 - T_K, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    mpc.set_nl_cons(expr_name = "T_K_upper", expr = T_K - 140.0, ub = 0.0, soft_constraint = True, penalty_term_cons=penalty_weight)
    
    # Input bounds
    mpc.bounds['lower', '_u', 'F'] = 5
    mpc.bounds['lower', '_u', 'Q_dot'] = -8500
    mpc.bounds['upper', '_u', 'F'] = 40
    mpc.bounds['upper', '_u', 'Q_dot'] = 0.0

    # Flag for regularization term
    mpc.flags["set_rterm"] = True

    # Parameter template for uncertain parameters
    p_template = mpc.get_p_template(1)
    mpc.set_p_fun(lambda t_now: p_template)

    # Prepare NLP problem
    mpc.prepare_nlp()

    # Add regularization term for control input changes
    u_prev = mpc.opt_p["_u_prev"]
    R = 1e2 * cd.diag([1e0, 1e0])  # Regularization matrix for the control input
    for idx in range(mpc.settings.n_horizon):
        u_now = mpc.opt_x_unscaled["_u", idx, 0]
        delta_u = u_now - u_prev
        delta_u = delta_u / (mpc._u_ub.master - mpc._u_lb.master)
        r_term = delta_u.T @ R @ delta_u  # Regularization term for the control input
        u_prev = u_now
        mpc._nlp_obj += r_term

    # Finalize NLP creation
    mpc.create_nlp()

    return mpc

# Main block to create RL_MPC instance
if __name__ == "__main__":
    rl_mpc = get_rl_mpc()