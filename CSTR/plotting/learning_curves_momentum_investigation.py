import os
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams.update({
    # "text.usetex": True,
    "text.usetex": False,
    "font.size": 12,
})

import pickle
from itertools import cycle

max_update = 100

dt = 0.005
def plot_learning_curve(beta_dict: dict[dict], figpath: str):

    if not os.path.exists(os.path.dirname(figpath)):
        os.makedirs(os.path.dirname(figpath))

    plt.switch_backend("agg")

    # ncols = 4
    ncols = 1
    nrows = len(beta_dict)

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols, 2 * nrows), constrained_layout=True, sharex=True, sharey=True, dpi=1200.0)

    max_update = 0

    for idx, (ikey, ivalue) in enumerate(beta_dict.items()):
        linestyles = cycle(["-", "--", "-.", ":"])
        colors = cycle(["tab:blue", "tab:orange", "tab:green", "tab:red"])

        ax[idx].set_title(ivalue["title"])

        for jdx, (jkey, jvalue) in enumerate(ivalue.items()):
            if jkey.startswith("eta"):
                linestyle = next(linestyles)
                color = next(colors)
                if "processed_results_list"in jvalue.keys():
                    mean_reward = [item["cum_reward"]["mean"] for item in jvalue["processed_results_list"]]
                    # mean_stage_cost = [-item["stage_cost"]["mean"] for item in jvalue["processed_results_list"]]
                    # mean_penalty = [-item["penalty"]["mean"] for item in jvalue["processed_results_list"]]
                    # mean_episode_time = [item["episode_time"]["mean"] for item in jvalue["processed_results_list"]]

                    ax[idx].plot(mean_reward, label=jvalue["label"], linestyle=linestyle, color = color)
                    max_update = 100


    ax[-1].set_xlabel(r"Iteration $k$")
    ax[-1].set_xlim([0, max_update])
    ax[-1].set_ylim([-110, -50])

    for axis in ax:
        axis.set_ylabel(r"$J(\theta_\alpha, \theta_\beta)$")
        axis.grid()
        axis.legend(loc="lower right", ncols=min(len(axis.get_lines()), 2), frameon=True)

    # handles, labels = ax[-1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 2), bbox_to_anchor=(0.5, -0.075), frameon=True)
    # plt.subplots_adjust(bottom=0.3)  # Make space for the legend

    plt.savefig(figpath, bbox_inches="tight")
    # plt.savefig(figpath)
    if not figpath.endswith(".pgf") and figpath.endswith(".png"):
        plt.savefig(figpath.replace(".png", ".pgf"), backend="pgf", bbox_inches="tight")
    elif not figpath.endswith(".pgf") and figpath.endswith(".pdf"):
        plt.savefig(figpath.replace(".pdf", ".pgf"), backend="pgf", bbox_inches="tight")
    else:
        print("Skipping saving as pgf, as the file is not a png oe pdf.")
    plt.close("all")
    return


if __name__ == "__main__":

    n_IC = 100
    omegainv = 1e1

    beta_list = [0.75, 0.9, 0.99]
    eta_list = [0.0, 0.5, 0.9, 0.99]
    
    beta_dict = {f"beta={beta:.3f}": {"beta": beta, "title": r"$\beta = " + f"{beta:.2f}$"} for beta in beta_list}

    for key, value in beta_dict.items():
        eta_dict = {}
        for jdx, eta in enumerate(eta_list):
            eta_dict[f"eta={eta:.3f}"] = {
                        "rl_method": "gauss_newton",
                        "actor_learning_rate": 1e-1,
                        "n_IC": n_IC,
                        "beta": value["beta"],
                        "eta": eta,
                        "omegainv": omegainv,
                        "label": r"$\eta = " + f"{eta:.2f}$",
                    }
        value.update(**eta_dict)
        beta_dict[key] = value
    
    for ikey, ivalue in beta_dict.items():
        for jkey, jvalue in ivalue.items():
            if jkey.startswith("eta"):
                rl_method = jvalue["rl_method"]
                actor_learning_rate = jvalue["actor_learning_rate"]
                n_IC = jvalue["n_IC"]
                beta = jvalue["beta"]
                eta = jvalue["eta"]
                omegainv = jvalue["omegainv"]

                result_path = os.path.join(
                    "CSTR",
                    "data",
                    rl_method,
                    f"lr{actor_learning_rate:.1e}_nIC{n_IC}_beta_{beta:.2f}_eta_{eta:.3f}_omegainv_{omegainv:.1e}",
                    "processed_results_list.pkl"
                    )

                if os.path.exists(result_path):
                    with open(result_path, "rb") as f:
                        processed_results_list = pickle.load(f)

                    beta_dict[ikey][jkey]["processed_results_list"] = processed_results_list


    figure_name = "momentum_learning_curve.png"
    figure_name = "momentum_learning_curve.pdf"
    figpath = os.path.join("CSTR", "data", "momentum_investigation", figure_name)

    plot_learning_curve(beta_dict, figpath = figpath)