# Computationally efficient Gauss-Newton reinforcement learning for model predictive control
This repository contains the implementation of the methods and results presented in the paper:
> **Computationally efficient Gauss-Newton reinforcement learning for model predictive control**  
> Dean Brandner, Sebastien Gros, and Sergio Lucia

## Overview
This project provides the necessary implementations for computationally efficient second-order reinforcement learning via Gauss-Newton Hessian approximations in application for model predictive control (MPC) policies.
This repository inludes:
1. An empirical validation of the superlinear convergence capability when the Gauss-Newton approximation is used in second-order reinforcement learning at the example of an analytical case study
2. Training routines of established first- and second-order reinforcement learning approaches with MPC policies for a nonlinear continuously stirred-tank reactor (CSTR) case study
3. Plotting utilitities to recreate the plots and evaluation tools to recreate results

## Repository structure
```
.
├── analytic_example/
│   └── analytic_cs.ipynb
├── CSTR/
│   ├── data/
│   │   └── results_will_appear_in_this_folder.txt
│   ├── helper/
│   │   └── steady_state_computation.py
│   ├── plotting/
│   │   ├── closed_loop_trajectories.ipynb
│   │   ├── comparison_to_sb3.ipynb
│   │   ├── dimensionality_investigation.ipynb
│   │   └── robustness_investigation.ipynb
│   └── training/
│       ├── RL_MPC.py
│       ├── closed_loop_trajectories.py
│       ├── compute_parametric_performance.py
│       ├── dimensionality_investigation_Adam.py
│       ├── dimensionality_investigation_approx_newton.py
│       ├── dimensionality_investigation_gauss_newton.py
│       ├── environments.py
│       ├── helper.py
│       ├── mp_utils.py
│       ├── mpc_collection.py
│       ├── rl_mpc_agents.py
│       ├── robustness_ic_investigation_Adam.py
│       ├── robustness_ic_investigation_approx_newton.py
│       ├── robustness_ic_investigation_gauss_newton.py
│       └── sb3_training_td3.py
├── .conda.yaml
├── .gitignore
├── .LICENSE.md
└── .README.md
```

## Key components
### Analytical case study
The results of the analytical case can be reproduced by running the jupyter notebook `analytic_cs.ipynb` in the folder `analytic_example`.

### CSTR case study
All training scripts are located under `CSTR/training`. All plotting utillities can be found under `CSTR/plotting`.
The results can be obtained in the following way.

#### Scalability investigation with respect to number of parameters
First run the following files:
- `dimensionality_investigation_Adam.py`
- `dimensionality_investigation_approx_newton.py`
- `dimensionality_investigation_gauss_newton.py`

Then plot the results by running `dimensionality_investigation.ipynb` in `CSTR/plotting`. The code will reproduce the results of Figure&nbsp;5 and Figure&nbsp;6. Note that the data and figure path may need to be adapted.

#### Influence of parameter scaling
First run the following files:
- `robustness_ic_investigation_Adam.py`
- `robustness_ic_investigation_approx_newton.py`
- `robustness_ic_investigation_gauss_newton.py`

Then plot the results by running `robustness_investigation.ipynb` in `CSTR/plotting`. The code will reproduce the results of Figure&nbsp;7 and Table&nbsp;2. Note that the data and figure path may need to be adapted.

#### Comparison to deep reinforcement learning
We assume that the influence of the parameter scaling is carried out beforehand.
To run the grid search or only the reported presented agent, you can run the following file:
- `sb3_training_td3.py`

Then plot the results by running `comparison_to_sb3.ipynb` in `CSTR/plotting`. The code will reproduce the results of Figure&nbsp;8 and Table&nbsp;4 if all experiments are carried out. Note that the data and figure path may need to be adapted.

#### Closed-loop trajectories
After all relevent experiments are performed, you can run `closed_loop_trajectories.py` to generate closed-loop trajectories. The closed-loop trajectories can be plotted by running `closed_loop_trajectories.ipynb` in `CSTR\plotting`.


## Usage
### Prerequisites
Clone the repository and install the [conda](https://www.anaconda.com/docs/getting-started/miniconda/install) environment that is stored in the file `.conda.yaml`

## License
See the LICENSE.md file for license rights and limitations.
