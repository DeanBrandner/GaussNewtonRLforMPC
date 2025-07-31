import os
from matplotlib import pyplot as plt

plt.rcParams.update({
    # "text.usetex": True,
    "text.usetex": False,
    "font.size": 14,
})

import numpy as np
import pickle
from itertools import cycle


def plot_points_into_2d_plot(parameters_dict: dict, alpha_mesh: np.ndarray, beta_mesh: np.ndarray, clc_mesh: np.ndarray, figpath: str):
    plt.switch_backend("agg")

    k = 1.0
    figsize = (k * 6.5, k * 5)

    fig, ax = plt.subplots(figsize=figsize, dpi=1200.0, constrained_layout=True)

    lower_cutoff = -125
    clc_mesh = np.clip(clc_mesh, a_min=lower_cutoff, a_max=None)
    cf00 = ax.contourf(alpha_mesh, beta_mesh, clc_mesh, levels=50, cmap='viridis')
    cf00.set_edgecolor('face')

    linestyles = cycle(["-", "--", "-.", ":"])
    markers = cycle(['o', 's', 'D', '*', 'v'])
    colors = cycle(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"])

    for method, method_data in parameters_dict.items():
        label = method_data["label"]
        linestyle = next(linestyles)
        marker = next(markers)
        color = next(colors)

        markersize = 5
        if marker == "*":
            markersize = 7
            

        ax.plot(
            method_data["parameters"][:, 0],
            method_data["parameters"][:, 1],
            linestyle=linestyle,
            marker=marker,
            color=color,
            markersize=markersize,
            label=label,
            markeredgecolor='black',
            markeredgewidth=0.5
        )
    pad = 0.0051  # same as colorbar pad
    cbar00 = fig.colorbar(cf00, ax=ax, orientation='vertical', pad=pad, location='right')
    # Access the colorbar axis and manually place the label
    cbar00.ax.text(
        0.5,                 # x in axis coords (0 left, 1 right)
        -0.04,
        r"$J(\theta_\alpha, \theta_\beta)$",
        ha='center',
        va='center',
        transform=cbar00.ax.transAxes
    )

    ticks = cbar00.get_ticks()
    ticklabels = [f"${tick:.0f}$" for tick in ticks]
    if len(ticklabels) > 0:
        ticklabels[0] = r"$\leq" + f"{lower_cutoff}" + "$"
        cbar00.set_ticks(ticks)
        cbar00.set_ticklabels(ticklabels)

    # for idx, label in enumerate(ticklabels)

    ax.set_xlabel(r"$\theta_{\alpha}$", labelpad=0)
    ax.set_ylabel(r"$\theta_{\beta}$")

    ax.set_xlim([np.min(alpha_mesh), np.max(alpha_mesh)])
    ax.set_ylim([np.min(beta_mesh), np.max(beta_mesh)])

    # fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.47, -0.1175))
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.15))

    fig.savefig(figpath, bbox_inches='tight')
    if not figpath.endswith(".pgf") and figpath.endswith(".png"):
        fig.savefig(figpath.replace(".png", ".pgf"), bbox_inches='tight')
    elif not figpath.endswith(".pgf") and figpath.endswith(".pdf"):
        fig.savefig(figpath.replace(".pdf", ".pgf"), bbox_inches='tight')
    plt.close("all")
    return


if __name__ == "__main__":

    method_dict = {
        "GA_Adam_1": {
            "rl_method": "GA_Adam",
            "actor_learning_rate": 1e-3,
            "n_IC": 100,
            "momentum": False,
            "adam": True,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "label": r"Adam $\alpha = 10^{-3}$",
        },
        "GA_Adam_2": {
            "rl_method": "GA_Adam",
            "actor_learning_rate": 1e-2,
            "n_IC": 100,
            "momentum": False,
            "adam": True,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "label": r"Adam $\alpha = 10^{-2}$",
        },
        "approx_newton": {
            "rl_method": "approx_newton",
            "actor_learning_rate": 1e-1,
            "n_IC": 100,
            "momentum": True,
            "adam": False,
            "beta": 0.75,
            "eta": 0.9,
            "omegainv":1e2,
            "label": r"Approx. Newton $\alpha = 0.1$",
        },
        "gauss_newton": {
            "rl_method": "gauss_newton",
            "actor_learning_rate": 1e-1,
            "n_IC": 100,
            "momentum": True,
            "adam": False,
            "beta": 0.75,
            "eta": 0.9,
            "omegainv":1e2,
            "label": r"Gauss-Newton $\alpha = 0.1$ (Proposed)",
        },
    }
        
    for method, params in method_dict.items():
        rl_method = params["rl_method"]
        actor_learning_rate = params["actor_learning_rate"]
        n_IC = params["n_IC"]
        adam = params["adam"]
        label = params["label"]

        if "Adam" in rl_method:
            rl_agent_path = os.path.join(
                "CSTR",
                "data",
                rl_method,
                f"lr{actor_learning_rate:.1e}_nIC{n_IC}_beta_1_{params["beta_1"]:.2f}_beta_2_{params["beta_2"]:.3f}"
                )
        else:
            rl_agent_path = os.path.join(
                "CSTR",
                "data",
                rl_method,
                f"lr{actor_learning_rate:.1e}_nIC{n_IC}_beta_{params["beta"]:.2f}_eta_{params["eta"]:.3f}_omegainv_{params["omegainv"]:.1e}"
            )
        
        parameter_dict = {}
        policy_gradient_dict = {}
        policy_hessian_dict = {}

        for item in os.listdir(rl_agent_path):
            if not "agent_update_" in item:
                continue

            with open(os.path.join(rl_agent_path, item, "rl_params.pkl"), "rb") as f:
                parameters = pickle.load(f)
            parameter_dict[item] = parameters.master
            
            if os.path.exists(os.path.join(rl_agent_path, item, "policy_gradients.pkl")):
                with open(os.path.join(rl_agent_path, item, "policy_gradients.pkl"), "rb") as f:
                    policy_gradient = pickle.load(f)
                policy_gradient_dict[item] = policy_gradient

            if os.path.exists(os.path.join(rl_agent_path, item, "policy_hessian.pkl")):
                with open(os.path.join(rl_agent_path, item, "policy_hessian.pkl"), "rb") as f:
                    policy_hessian = pickle.load(f)
                policy_hessian_dict[item] = policy_hessian

            
        parameter_list = []
        policy_gradient_list = []
        policy_hessian_list = []
        for idx in range(len(parameter_dict)):
            parameter_list.append(parameter_dict[f"agent_update_{idx}"])
            if f"agent_update_{idx}" in policy_gradient_dict:
                policy_gradient_list.append(policy_gradient_dict[f"agent_update_{idx}"])
            if f"agent_update_{idx}" in policy_hessian_dict:
                policy_hessian_list.append(policy_hessian_dict[f"agent_update_{idx}"])

        parameters = np.hstack(parameter_list).T
        policy_gradients = np.hstack(policy_gradient_list).T
        if len(policy_hessian_list) > 0:
            policy_hessians = np.stack(policy_hessian_list)
        else:
            policy_hessians = None

        method_dict[method]["parameters"] = parameters
        method_dict[method]["policy_gradients"] = policy_gradients
        method_dict[method]["policy_hessians"] = policy_hessians

    parametric_results_base_path = os.path.join("CSTR", "data", "parametric_results", f"n_initial_conditions_{n_IC}") 
    with open(os.path.join(parametric_results_base_path, "param_range_alpha_mesh.pkl"), "rb") as f:
        alpha_mesh = pickle.load(f)
    with open(os.path.join(parametric_results_base_path, "param_range_beta_mesh.pkl"), "rb") as f:
        beta_mesh = pickle.load(f)
    with open(os.path.join(parametric_results_base_path, "processed_results_mesh.pkl"), "rb") as f:
        clc_mesh = pickle.load(f)
    
    clc_mean_mesh = np.zeros(clc_mesh.shape)
    for idx in range(clc_mesh.shape[0]):
        for jdx in range(clc_mesh.shape[1]):
            clc_mean_mesh[idx, jdx] = clc_mesh[idx, jdx]["return"]["mean"]

    figname = "param_trajectories_2d.pdf"
    figpath = os.path.join("CSTR", "data", "parameter_plots_2d", figname)
    plot_points_into_2d_plot(method_dict, alpha_mesh, beta_mesh, clc_mean_mesh, figpath = figpath)