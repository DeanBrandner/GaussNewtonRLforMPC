import os
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({
    # "text.usetex": True,
    "text.usetex": False,
    "font.size": 12,
    # "text.latex.preamble": r"\usepackage{amsmath}",
})

import pickle

def plot_hessian_with_momentum(policy_hessian: np.ndarray, figpath: str, eta:float = 0.9, omega_inv:float = 10):
    
    B_init = -np.eye(policy_hessian.shape[1]) * omega_inv / (1 - eta)
    D_init = B_init * (1 - eta)
    D = D_init.copy()
    D_hat_init = D_init / (1 - eta)

    B_list = [B_init.copy()]
    D_list = [D_init.copy()]
    D_hat_list = [D_hat_init.copy()]

    eigenvalues_B_list = []
    eigenvalues_D_list = []
    eigenvalues_D_hat_list = []

    for idx in range(policy_hessian.shape[0]):
        B = policy_hessian[idx, :, :].copy()
        D = eta * D + (1 - eta) * B

        D_hat = D / (1 - eta ** (idx + 1 + 1))


        eigenvalues_B = np.linalg.eigh(B)[0]
        eigenvalues_D = np.linalg.eigh(D)[0]
        eigenvalues_D_hat = np.linalg.eigh(D_hat)[0]

        B_list.append(B)
        D_list.append(D)
        D_hat_list.append(D_hat)
        eigenvalues_B_list.append(eigenvalues_B)
        eigenvalues_D_list.append(eigenvalues_D)
        eigenvalues_D_hat_list.append(eigenvalues_D_hat)

    B = np.stack(B_list)
    D = np.stack(D_list)
    D_hat = np.stack(D_hat_list)

    eigenvalues_B = np.stack(eigenvalues_B_list)
    eigenvalues_D = np.stack(eigenvalues_D_list)
    eigenvalues_D_hat = np.stack(eigenvalues_D_hat_list)
    
    plt.switch_backend("agg")

    ncols = 1
    nrows = 2

    scaling_factor = 1.0
    
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = (5 * ncols * scaling_factor, 2 * nrows  * scaling_factor), constrained_layout = True, sharex=True)

    label_hessian = r"$\tilde{\boldsymbol{B}}$"
    label_hessian_D = r"$\boldsymbol{D}$"
    label_hessian_D_hat = r"$\hat{\boldsymbol{D}}$"


    ax[0].plot(-eigenvalues_B[:, 0], label = label_hessian, linestyle = "solid")
    ax[0].plot(-eigenvalues_D[:, 0], label = label_hessian_D, linestyle = "dashed")
    ax[0].plot(-eigenvalues_D_hat[:, 0], label = label_hessian_D_hat, linestyle = "dashdot")
    ax[0].set_title("First eigenvalue")
    ax[0].set_ylabel(r"$-\lambda_1$")

    ax[1].plot(-eigenvalues_B[:, 1], label = label_hessian, linestyle = "solid")
    ax[1].plot(-eigenvalues_D[:, 1], label = label_hessian_D, linestyle = "dashed")
    ax[1].plot(-eigenvalues_D_hat[:, 1], label = label_hessian_D_hat, linestyle = "dashdot")
    ax[1].set_title("Second eigenvalue")
    ax[1].set_ylabel(r"$-\lambda_2$")


    ax[-1].set_xlabel(r"Iteration $k$")
    for axis in ax.flatten():
        axis.grid()
        axis.set_xlim([0, policy_hessian.shape[0] - 1])
        axis.set_yscale("log")
        axis.legend(loc="lower right", ncol=3)

    plt.savefig(figpath, dpi =1200.0, bbox_inches='tight')
    plt.close("all")
    return

if __name__ == "__main__":
    rl_method = "gauss_newton" # "gauss_newton", "approx_newton",

    actor_learning_rate = 1e-1
    use_momentum = True
    use_adam = False
    n_IC = 100


    beta = 0.75
    eta = 0.90
    omegainv = 1e1
    rl_agent_path = os.path.join("CSTR", "data", rl_method, f"lr{actor_learning_rate:.1e}_nIC{n_IC}_beta_{beta:.2f}_eta_{eta:.3f}_omegainv_{omegainv:.1e}")



    # Loading
    policy_hessian_dict = {}

    for item in os.listdir(rl_agent_path):
        if not "agent_update_" in item:
            continue
        
        if os.path.exists(os.path.join(rl_agent_path, item, "policy_hessian.pkl")):
            with open(os.path.join(rl_agent_path, item, "policy_hessian.pkl"), "rb") as f:
                policy_hessian = pickle.load(f)
            policy_hessian_dict[item] = policy_hessian

    policy_hessian_list = []
    for idx in range(len(policy_hessian_dict)):
        if f"agent_update_{idx}" in policy_hessian_dict:
            policy_hessian_list.append(policy_hessian_dict[f"agent_update_{idx}"])

    if len(policy_hessian_list) > 0:
        policy_hessians = np.stack(policy_hessian_list)
    else:
        policy_hessians = None


    # Plotting
    figpath = os.path.join(rl_agent_path, "smoothed_hessians.png")
    plot_hessian_with_momentum(policy_hessians, figpath = figpath)
