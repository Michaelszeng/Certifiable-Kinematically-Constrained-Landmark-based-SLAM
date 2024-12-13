# Certifiably-Optimal Kinematically-Constrained Landmark-based SLAM

We present a certifiably optimal algorithm to solve Kinematically-Constrained Landmark-based SLAM. 

See our paper for more details.

**This codebase includes 4 different solvers**:
 - `cvxpy`: Solves the "Separated linear terms" formulation as described in our paper. This implementation is the most lightweight, directly encoding the SDP cost and constraint matrices.
 - `drake1`: Solves the "Separated linear terms" formulation as described in our paper. Implemented using `drake` toolbox, allowing much simpler problem-definition in least-squares form and automatic generation of the SDP cost and constraint matrices at the expense of more overhead.
 - `drake2`: Solves the SDP as described in our paper.
 - `nonlinear`: Solves the non-convex least squares optimization using SNOPT.


## Installation

It is recommended that you create a Python virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

Install the required dependencies:
```
pip install -r requirements.txt
```

#### MOSEK License

Without a MOSEK license, you will only be able to run the nonlinear solver (which is not a certifiable solver as described in our paper).

If you are part of an academic institution, you can obtain a license [here](https://www.mosek.com/products/academic-licenses/). 

To make your MOSEK license visible to `cvxpy` (used in `solver_cvxpy.py`), place the license file in `~/mosek/mosek.lic`.

To make your MOSEK license visible to `drake` (used in `solver_drake1.py` and `solver_drake2.py`), create an environment variable "MOSEKLM_LICENSE_FILE" and set it to the path to your license file.


## Usage
To run our code:

    python main.py

This codebase includes multiple solvers for multiple formulations of the SDP, and many ways to generate test scenarios. To make these selections, modify the options at the top of `main.py`.


## Benchmarking

The benchmarking code is located in the `benchmarking` folder. To begin, run one of the following scripts: `benchmark_error.py`, `benchmark_rank.py`, or `benchmark_runtime.py`. These scripts will generate the necessary data and save it as `.npy` files. For running custom trials, you can edit the `trials` list within each script, using the provided example as a guide.

After generating the data, use the corresponding graphing script, `graph_error.py`, `graph_rank.py`, or `graph_runtime.py`, to process the `.npy` files and produce the graphs. For custom trials, remember to update the `files` list in the graphing scripts to reflect the new file names.

