import os
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({
    # "text.usetex": True,
    "text.usetex": False,
    "font.size": 14,
})

from tqdm import tqdm
import pickle
from itertools import cycle, product

max_update = 100



def plot_learning_curve(processed_results_dict: dict[dict], figpath: str, plot_in_subfigures: bool = False, plot_average:bool = False):

    # Preprocessing based on average or not
    preprocessed_dict = {}
    for method, initial_guesses in processed_results_dict.items():
        preprocessed_dict[method] = {}
        for initial_guess, results in initial_guesses.items():
            preprocessed_dict[method][initial_guess] = {
                "label": results["label"],
                }
            preprocessed_dict[method]["title"] = results["title"]
            if "processed_results_list" in results:
                processed_results = np.array([item["cum_reward"]["mean"] for item in results["processed_results_list"]])
                preprocessed_dict[method][initial_guess]["processed_results"] = processed_results
    
    if plot_average:
        for method, initial_guesses in preprocessed_dict.items():
            stacked_results = [item["processed_results"] for item in initial_guesses.values() if "processed_results" in item]
            if len(stacked_results) > 0:
                max_length = max(len(item) for item in stacked_results)
                stacked_results = [item for item in stacked_results if len(item) == max_length]
                stacked_results = np.stack(stacked_results)

                preprocessed_dict[method]["mean"] = stacked_results.mean(axis=0)
                preprocessed_dict[method]["std"] = stacked_results.std(axis=0)
                preprocessed_dict[method]["median"] = np.median(stacked_results, axis=0)
                preprocessed_dict[method]["min"] = stacked_results.min(axis=0)
                preprocessed_dict[method]["max"] = stacked_results.max(axis=0)

    if not os.path.exists(os.path.dirname(figpath)):
        os.makedirs(os.path.dirname(figpath))

    plt.switch_backend("agg")

    nrows = 1
    ncols = 1
    k = 0.4
    figsize = (16 * k * ncols, 9 * k * nrows)    
    if plot_in_subfigures:
        nrows = len(processed_results_dict)
        figsize = (5 * ncols, 3 * nrows)    
    alpha = 0.2


    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, constrained_layout=True, sharex=True, sharey= True)

    max_update = 0

    colors = cycle(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"])
    linestyles = cycle(["-", "--", "-.", ":"])

    if plot_in_subfigures:
        for idx, (method, initial_guesses) in enumerate(preprocessed_dict.items()):
            ax[idx].set_title(initial_guesses["title"])

            if plot_average:
                if "mean" in initial_guesses.keys():
                    ax[idx].plot(initial_guesses["mean"], label="Mean", linestyle="-")
                    ax[idx].fill_between(np.arange(len(initial_guesses["mean"])), 
                                         initial_guesses["mean"] - initial_guesses["std"], 
                                         initial_guesses["mean"] + initial_guesses["std"], 
                                         alpha=alpha, label=r"$\pm$ Std")
                    max_update = max(max_update, len(initial_guesses["mean"]) - 1)

            else:
                colors = cycle(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"])
                linestyles = cycle(["-", "--", "-.", ":"])
                for initial_guess, results in initial_guesses.items():
                    if not "initial_guess" in initial_guess:
                        continue
                    label = results["label"]

                    linestyle = next(linestyles)
                    color = next(colors)

                    if "processed_results" in results:
                        reward = results["processed_results"]
                        ax[idx].plot(reward, label=label, linestyle=linestyle, color=color)
                        max_update = max(max_update, len(reward) - 1)


            for axis in ax:
                axis.grid()
                axis.set_xlabel(r"Iteration $k$")
                axis.set_ylabel(r"$J(\theta_\alpha, \theta_\beta)$")

        if plot_average:
            handles, labels = ax[-1].get_legend_handles_labels()
            # Place the legend below the plots, inside the figure area
            fig.legend(handles, labels, loc="lower center", ncol=min(max(1, len(handles)), 5), bbox_to_anchor=(0.525, -0.04), frameon=True)
        else:
            handles, labels = ax[-1].get_legend_handles_labels()
            # Place the legend below the plots, inside the figure area
            fig.legend(handles, labels, loc="lower center", ncol=min(max(1, len(handles)), 5), bbox_to_anchor=(0.5, -0.075), frameon=True)

        tuning_axis = ax[-1]
        tuning_axis.set_xlim([0, max_update])
        tuning_axis.set_ylim([-100, -50])
    

    else:
        for idx, (method, initial_guesses) in enumerate(preprocessed_dict.items()):
            if plot_average:
                color = next(colors)
                linestyle = next(linestyles)

                if "mean" in initial_guesses.keys():
                    label = initial_guesses["title"]
                    ax.plot(initial_guesses["mean"], label=label, linestyle=linestyle, color=color)
                    ax.fill_between(
                        np.arange(len(initial_guesses["mean"])),
                        initial_guesses["mean"] - initial_guesses["std"],
                        initial_guesses["mean"] + initial_guesses["std"],
                        alpha=alpha,
                        # label=label + r" $\pm$ Std.",
                        color=color,
                        linewidth=0  # Remove border/frame
                    )
                    max_update = max(max_update, len(initial_guesses["mean"]) - 1)

            else:
                linestyle = next(linestyles)
                color = next(colors)
                label = initial_guesses["title"]
                label_printed = False
                for initial_guess, results in initial_guesses.items():                    

                    if "processed_results" in results:
                        reward = results["processed_results"]
                        if not label_printed:
                            ax.plot(reward, label=label, linestyle = linestyle, color = color)
                            label_printed = True
                        else:
                            ax.plot(reward, linestyle = linestyle, color = color)
                        max_update = max(max_update, len(reward) - 1)

            ax.grid()
            ax.set_xlabel(r"Iteration $k$")
            ax.set_ylabel(r"$J(\theta_\alpha, \theta_\beta)$")

        if plot_average:
            ax.legend(loc = "lower right")
        else:
            handles, labels = ax.get_legend_handles_labels()
            # Place the legend below the plots, inside the figure area
            fig.legend(handles, labels, loc="lower center", ncol=min(max(1, len(handles)), 2), bbox_to_anchor=(0.5, -0.1), frameon=True)

        ax.set_xlim([0, max_update])
        ax.set_ylim([-100, -50])


    plt.savefig(figpath, bbox_inches="tight", dpi = 1200.0)
    plt.close("all")
    return



if __name__ == "__main__":

    basepath = os.path.join("CSTR", "data", "ic_investigation")

    nIC = 100

    figformat_list = ["png", "pdf"]

    n_initial_guesses = 5

    # Load the agents trained with different algorithms

    # Gauss-Newton
    lr_gauss_newton = 1e-1
    beta_gauss_newton = 0.75
    eta_gauss_newton = 0.9
    omegainv_gauss_newton = 1e2

    gauss_newton_dict = {}
    for idx_initial_guess in tqdm(iterable=range(n_initial_guesses), desc="Loading results for Gauss-Newton"):
        initial_guess_dict =  {
            "rl_method": "gauss_newton",
            "actor_learning_rate": lr_gauss_newton,
            "beta": beta_gauss_newton,
            "eta": eta_gauss_newton,
            "omegainv": omegainv_gauss_newton,
            "n_IC": nIC,
            "momentum": True,
            "adam": False,
            "label": r"$i = " + f"{idx_initial_guess + 1}$",
            "title": r"Gauss-Newton $\alpha = " + f"{lr_gauss_newton:.1f}$ (Proposed)",
        }

        data_path = os.path.join(
            basepath,
            initial_guess_dict["rl_method"],
            f"lr{initial_guess_dict['actor_learning_rate']:.1e}_nIC{initial_guess_dict['n_IC']}_beta_{initial_guess_dict['beta']:.2f}_eta_{initial_guess_dict['eta']:.3f}_omegainv_{initial_guess_dict['omegainv']:.1e}",
            f"initial_guess_{idx_initial_guess}",
            "processed_results_list.pkl"
        )
        if os.path.exists(data_path):
            with open(data_path, "rb") as f:
                processed_results = pickle.load(f)
            initial_guess_dict["processed_results_list"] = processed_results

        gauss_newton_dict[f"initial_guess_{idx_initial_guess + 1}"] = initial_guess_dict


    # Approximate Newton
    lr_approximate_newton = 1e-1
    beta_approximate_newton = 0.75
    eta_approximate_newton = 0.9
    omegainv_approximate_newton = 1e2

    approximate_newton_dict = {}
    for idx_initial_guess in tqdm(iterable=range(n_initial_guesses), desc="Loading results for Approximate Newton"):
        initial_guess_dict =  {
            "rl_method": "approx_newton",
            "actor_learning_rate":lr_approximate_newton,
            "beta": beta_approximate_newton,
            "eta": eta_approximate_newton,
            "omegainv": omegainv_approximate_newton,
            "n_IC": nIC,
            "momentum": True,
            "adam": False,
            "label": r"$i = " + f"{idx_initial_guess + 1}$",
            "title": r"Approx. Newton $\alpha = " + f"{lr_approximate_newton:.1f}$",
        }

        data_path = os.path.join(
            basepath,
            initial_guess_dict["rl_method"],
            f"lr{initial_guess_dict['actor_learning_rate']:.1e}_nIC{initial_guess_dict['n_IC']}_beta_{initial_guess_dict['beta']:.2f}_eta_{initial_guess_dict['eta']:.3f}_omegainv_{initial_guess_dict['omegainv']:.1e}",
            f"initial_guess_{idx_initial_guess}",
            "processed_results_list.pkl"
        )
        if os.path.exists(data_path):
            with open(data_path, "rb") as f:
                processed_results = pickle.load(f)
            initial_guess_dict["processed_results_list"] = processed_results

        approximate_newton_dict[f"initial_guess_{idx_initial_guess + 1}"] = initial_guess_dict


    # Adam
    momentum_type = "Adam"
    lr_adam = 1e-3
    adam_beta_1 = 0.9
    adam_beta_2 = 0.999

    adam_dict = {}
    for idx_initial_guess in tqdm(iterable=range(n_initial_guesses), desc=f"Loading results for Gradient ascent with {momentum_type}"):
        initial_guess_dict =  {
            "rl_method": f"GA_{momentum_type}",
            "actor_learning_rate": lr_adam,
            "n_IC": nIC,
            "momentum": False,
            "adam": True,
            "adam_beta_1": adam_beta_1,
            "adam_beta_2": adam_beta_2,
            "label": r"$i = " + f"{idx_initial_guess + 1}$",
            "title": r"Adam $\alpha = 10^{-3}$",
        }

        data_path = os.path.join(
            basepath,
            initial_guess_dict["rl_method"],
            f"lr{initial_guess_dict['actor_learning_rate']:.1e}_nIC{initial_guess_dict['n_IC']}_beta_1_{adam_beta_1:.2f}_beta_2_{adam_beta_2:.3f}",
            f"initial_guess_{idx_initial_guess}",
            "processed_results_list.pkl"
        )
        if os.path.exists(data_path):
            with open(data_path, "rb") as f:
                processed_results = pickle.load(f)
            initial_guess_dict["processed_results_list"] = processed_results

        adam_dict[f"initial_guess_{idx_initial_guess + 1}"] = initial_guess_dict



    # Combine all methods into a single dictionary
    method_dict = {
        "Adam": adam_dict,
        "approximate_newton": approximate_newton_dict,
        "gauss_newton": gauss_newton_dict,
    }   


    # Do the four plots
    plot_in_subfigures_list = [True, False]
    plot_average_list = [False, True]

    for figformat in figformat_list:
        print(f"Plotting with figformat={figformat}")
        iterates = product(plot_in_subfigures_list, plot_average_list)
        for plot_in_subfigures, plot_average in iterates:
            print(f"Plotting with plot_in_subfigures={plot_in_subfigures} and plot_average={plot_average}")
            if plot_average and plot_in_subfigures:
                figure_name = f"learning_curve_ic_investigation_sf_avg." + figformat
            elif plot_average and not plot_in_subfigures:
                figure_name = f"learning_curve_ic_investigation_avg." + figformat
            elif not plot_average and plot_in_subfigures:
                figure_name = f"learning_curve_ic_investigation_sf." + figformat
            else:
                figure_name = f"learning_curve_ic_investigation." + figformat
            
            figpath = os.path.join("CSTR", "data", "ic_investigation", "figs", figure_name)
            plot_learning_curve(method_dict, figpath = figpath, plot_in_subfigures=plot_in_subfigures, plot_average=plot_average)