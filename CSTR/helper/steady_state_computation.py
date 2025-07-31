import casadi as cd
import numpy as np

# Set random seed and parameter statistics
seed = 11
alpha_mean = 1.0
alpha_std = 0.5e-1
beta_mean = 1.0
beta_std = 0.5e-1
alpha_max = 1.15
alpha_min = 0.85
beta_max = 1.15
beta_min = 0.85

def normal_pdf(x, mu=0, sigma=1):
    """
    Computes the probability density function of a normal distribution.
    Parameters:
        x (float or array-like): The point(s) at which to evaluate the PDF.
        mu (float): The mean of the distribution.
        sigma (float): The standard deviation of the distribution.
    Returns:
        float or array-like: The value(s) of the PDF at x.
    """
    coeff = 1 / ((2 * cd.pi * sigma **2) ** 0.5)
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * cd.exp(exponent)

def compute_steady_state_wrt_x(n_samples:int = 1):
    """
    Computes the steady-state solution for a CSTR reactor model using CasADi and IPOPT.
    Parameters:
        n_samples (int): Number of samples for uncertain parameters alpha and beta.
    Returns:
        SP_opt: Steady-state temperatures (reactor and jacket).
    """

    # --- Physical and model constants ---
    K0_ab = 1.287e12 # Pre-exponential factor [h^-1]
    K0_bc = 1.287e12 # Pre-exponential factor [h^-1]
    K0_ad = 9.043e9  # Pre-exponential factor [l/mol.h]
    R_gas = 8.3144621e-3 # Universal gas constant [kj/mol.K]
    E_A_ab = 9758.3*1.00 # Activation energy [kj/mol]
    E_A_bc = 9758.3*1.00 # Activation energy [kj/mol]
    E_A_ad = 8560.0*1.0  # Activation energy [kj/mol]
    H_R_ab = 4.2         # Reaction enthalpy [kj/mol A]
    H_R_bc = -11.0       # Reaction enthalpy [kj/mol B] Exothermic
    H_R_ad = -41.85      # Reaction enthalpy [kj/mol A] Exothermic
    Rou = 0.9342         # Density [kg/l]
    Cp = 3.01            # Specific Heat capacity [kj/Kg.K]
    Cp_k = 2.0           # Coolant heat capacity [kj/kg.k]
    A_R = 0.215          # Area of reactor wall [m^2]
    V_R = 10.01          # Volume of reactor [l]
    m_k = 5.0            # Coolant mass[kg]
    T_in = 130.0         # Temp of inflow [Celsius]
    K_w = 4032.0         # Heat transfer coefficient [kj/h.m^2.K]
    C_A0 = (5.7+4.5)/2.0*1.0 # Initial concentration of A [mol/l]

    # --- Sample uncertain parameters alpha and beta ---
    alpha_list = []
    beta_list = []
    alpha_num = []
    beta_num = []
    rng = np.random.default_rng(seed)
    for i in range(n_samples):
        # Symbolic variables for alpha and beta
        alpha_list.append(cd.SX.sym(f"alpha_{i}"))
        beta_list.append(cd.SX.sym(f"beta_{i}"))
        # Sampled numeric values for alpha and beta (clipped to bounds)
        alpha_num.append(np.clip(rng.normal(loc = alpha_mean, scale = alpha_std), a_max = alpha_max, a_min = alpha_min))
        beta_num.append(np.clip(rng.normal(loc = beta_mean, scale = beta_std), a_max = beta_max, a_min = beta_min))

    # --- Lists to collect optimization variables, bounds, and initial guesses ---
    x_var_list = []  # State variables
    xlb_list = []    # Lower bounds for states
    xub_list = []    # Upper bounds for states
    x_init = []      # Initial guess for states

    u_var_list = []  # Control variables
    ulb = []         # Lower bounds for controls
    uub = []         # Upper bounds for controls
    u_init = []      # Initial guess for controls

    g_list = []      # Constraints
    glb_list = []    # Lower bounds for constraints
    gub_list = []    # Upper bounds for constraints

    obj = 0.0        # Objective function

    SP_list = []     # List to store setpoints (temperatures)

    # --- Build optimization problem for each alpha/beta sample pair ---
    for idx, alpha in enumerate(alpha_list):
        for jdx, beta in enumerate(beta_list):
            # --- Define state variables ---
            C_a = cd.SX.sym(f"C_a_alpha{idx}_beta{jdx}")   # Concentration of A
            C_b = cd.SX.sym(f"C_b_alpha{idx}_beta{jdx}")   # Concentration of B
            T_R = cd.SX.sym(f"T_R_alpha{idx}_beta{jdx}")   # Reactor temperature
            T_K = cd.SX.sym(f"T_K_alpha{idx}_beta{jdx}")   # Jacket temperature

            x = cd.vertcat(C_a, C_b, T_R, T_K)
            x_var_list.append(x)
            SP_list.append(cd.vertcat(T_R, T_K))

            # --- State bounds and initial guesses ---
            xlb_list.extend([0.1, 0.1, 50.0, 50.0])      # Lower bounds
            xub_list.extend([2.0, 2.0, 140.0, 140.0])    # Upper bounds
            x_init.extend([1.0, 1.0, 100.0, 100.0])      # Initial guess

            # --- Define control variables ---
            F = cd.SX.sym(f"F_alpha{idx}_beta{jdx}")         # Flow rate
            Q_dot = cd.SX.sym(f"Q_dot_alpha{idx}_beta{jdx}") # Heat input

            u = cd.vertcat(F, Q_dot)
            u_var_list.append(u)

            # --- Control bounds and initial guesses ---
            ulb.extend([5, -8.50])      # Lower bounds
            uub.extend([40, 0.0])       # Upper bounds
            u_init.extend([25, -2.0])   # Initial guess

            # --- Scaling for Q_dot ---
            Q_dot = 1e3 * Q_dot

            # --- Reaction rate expressions ---
            K_1 = beta * K0_ab * cd.exp((-E_A_ab)/((T_R+273.15)))
            K_2 =  K0_bc * cd.exp((-E_A_bc)/((T_R+273.15)))
            K_3 = K0_ad * cd.exp((-alpha*E_A_ad)/((T_R+273.15)))

            T_dif = T_R - T_K

            # --- Differential equations (steady-state: set to zero) ---
            dCa_dt = F * (C_A0 - C_a) - K_1 * C_a - K_3 * (C_a ** 2)
            dCb_dt = - F * C_b + K_1 * C_a - K_2 * C_b
            dTR_dt = ((K_1*C_a*H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2)*H_R_ad)/(-Rou*Cp)) + F*(T_in-T_R) +(((K_w*A_R)*(-T_dif))/(Rou*Cp*V_R))
            dTK_dt = (Q_dot + K_w*A_R*(T_dif))/(m_k*Cp_k)

            dstates_dt = cd.vertcat(dCa_dt, dCb_dt, dTR_dt, dTK_dt)
            g_list.append(dstates_dt)
            glb_list.append(cd.DM.zeros(dstates_dt.shape)) # Lower bounds for constraints
            gub_list.append(cd.DM.zeros(dstates_dt.shape)) # Upper bounds for constraints

            # --- Objective function (maximize B, penalize C and D) ---
            C_c = K_2 / F * C_b
            C_d = K_3 / F * C_a ** 2
            obj += - (100 * C_b * F) + (10 * C_c * F) + (5 * C_d * F)

    # --- Normalize objective by number of samples ---
    obj /= n_samples**2

    # --- Add constraints to enforce identical setpoints across samples ---
    for SP1, SP2 in zip(SP_list[:-1], SP_list[1:]):
        g_list.append(SP1 - SP2)
        glb_list.append(cd.DM.zeros(SP1.shape))
        gub_list.append(cd.DM.zeros(SP1.shape))

    # --- Stack all optimization variables, bounds, and constraints ---
    optimization_variables = cd.vertcat(*x_var_list, *u_var_list)
    optimization_variables_inv_func = cd.Function("optimization_variables_inv_func", [optimization_variables], [SP_list[0]], ["optimization_variables"], ["SP"])

    optimization_variables_lb = cd.vertcat(*xlb_list, *ulb)
    optimization_variables_ub = cd.vertcat(*xub_list, *uub)
    optimization_variabels_init = cd.vertcat(*x_init, *u_init)

    g = cd.vertcat(*g_list)
    glb = cd.vertcat(*glb_list)
    gub = cd.vertcat(*gub_list)

    # --- Stack parameters and their numeric values ---
    parameters = cd.vertcat(*alpha_list, *beta_list)
    parameters_num = cd.vertcat(*alpha_num, *beta_num)

    # --- Define NLP problem dictionary for CasADi ---
    nlp_dict = {
        "x": optimization_variables,
        "f": obj,
        "g": g,
        "p": parameters
    }

    # --- Create NLP solver (IPOPT) and solve ---
    steady_state_solver = cd.nlpsol("steady_state_solver", "ipopt", nlp_dict)

    res = steady_state_solver(
        x0=optimization_variabels_init,
        p = parameters_num,
        lbg=glb,
        ubg=gub,
        lbx=optimization_variables_lb,
        ubx=optimization_variables_ub,
    )

    # --- Extract optimal setpoint temperatures ---
    SP_opt = optimization_variables_inv_func(res["x"])

    return SP_opt

if __name__ == "__main__":
    # Run steady-state computation for 25 samples
    x_ss = compute_steady_state_wrt_x(25)
    print("Steady state computation completed.")
    print(f"Temperatures at steady state:")
    print(f"Reactor: \t{float(x_ss[0]):.0f} C")
    print(f"Jacket: \t{float(x_ss[1]):.0f} C")