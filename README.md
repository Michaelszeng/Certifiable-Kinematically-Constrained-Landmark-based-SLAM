# Certifiably-Optimal Kinematically Constrained Landmark-based SLAM

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

## Usage

### CVXPY Implementation



**Note:** If you are using MOSEK, you need to put the MOSEK license (obtained [here](https://www.mosek.com/products/academic-licenses/)) in `~/mosek/mosek.lic`.

### Drake Implementation

**Note:** Mosek license needed. Also, Drake Mosek installation isn't trivial.

## Benchmarking

The benchmarking code is located in the `benchmarking` folder. To begin, run one of the following scripts: `benchmark_error.py`, `benchmark_rank.py`, or `benchmark_runtime.py`. These scripts will generate the necessary data and save it as `.npy` files. For running custom trials, you can edit the `trials` list within each script, using the provided example as a guide.

After generating the data, use the corresponding graphing script, `graph_error.py`, `graph_rank.py`, or `graph_runtime.py`, to process the `.npy` files and produce the graphs. For custom trials, remember to update the `files` list in the graphing scripts to reflect the new file names.

