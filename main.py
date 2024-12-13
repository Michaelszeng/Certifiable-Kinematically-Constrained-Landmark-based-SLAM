import numpy as np
np.set_printoptions(edgeitems=30, linewidth=270, precision=4, suppress=True)
from scipy.spatial.transform import Rotation   
import os
import sys
import time
from enum import Enum

from solver_utils import *
from visualization_utils import visualize_results
from evaluation import compute_relaxation_gap, compute_mean_errors

class Solver(Enum):
    cvxpy = 1
    drake1 = 2
    drake2 = 3
    snopt = 4


SOLVER = Solver.drake2  # Select a solver from the Solver enums

PRESET_TEST = None  # Set to None to generate a test case with random landmark locations, or set to a specific test file to run that test case

MEASUREMENT_NOISE = 0.01         # Standard deviation of Gaussian noise added to measurements
MEASUREMENT_DROPOUT = 0.01       # Probability of measurements being dropped (to simulate occlusions or object detection failures)

# Manually define the ground truth trajectory
if PRESET_TEST is None:
    # Example: Spiral trajectory
    velocity_trajectory = np.array([1, 0, 0.5])
    angular_velocity_trajectory = np.array([0, 0, 45])  # RPY angular velocity in degrees
    K = 4  # Number of landmarks
    N = 4  # Number of timesteps
    d = 3  # Number of dimensions of space

    # Example: Snake trajectory
    #v_gt = np.array([1, 0, 0])
    #true_rpy_vel = np.array([
    #    [0, 0, 0],
    #    [0, 0, 10],
    #    [0, 0, 20],
    #    [0, 0, 10],
    #    [0, 0, -10],
    #    [0, 0, -20],
    #    [0, 0, -10],
    #])  # RPY angular velocity in degrees
    #K = 4  # Number of landmarks
    #N = 8  # Number of timesteps
    
    # Velocity, angular velocity, and measurement covariances
    cov_v = 1
    cov_omega = 1
    cov_meas = 1

    # Define ranges for each dimension
    x_range = (-10, 10)
    y_range = (-10, 10)
    z_range = (-10, 10)
    
    x_coords = np.random.uniform(x_range[0], x_range[1], size=K)
    y_coords = np.random.uniform(y_range[0], y_range[1], size=K)
    z_coords = np.random.uniform(z_range[0], z_range[1], size=K)
    p_gt = np.stack((x_coords, y_coords, z_coords), axis=1)

    Omega_gt = Rotation.from_euler("xyz", angular_velocity_trajectory, degrees=True).as_matrix()

    v_gt = velocity_trajectory
    x_gt, R_gt = generate_ground_truth(N, v_gt, Omega_gt)
    print_ground_truth(Omega_gt, R_gt, p_gt, v_gt, x_gt)

    y_bar = generate_measurements(x_gt, R_gt, p_gt, noise=MEASUREMENT_NOISE, dropout=MEASUREMENT_DROPOUT)
        
else:
    # Dynamically import the specified test file
    current_folder = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(current_folder, "test_data")
    sys.path.append(test_data_path)


# Import the correct solver
if SOLVER == Solver.cvxpy:
    from solver_cvxpy import solver
    d = 3  # CVXPY solver currently is programmed for 3D postions/rotations only
elif SOLVER == Solver.drake1:
    from solver_drake1 import solver
elif SOLVER == Solver.drake2:
    from solver_drake2 import solver
elif SOLVER == Solver.snopt:    
    from solver_snopt import solver

Omega, R, p, v, x, rank, S = solver(y_bar, N, K, d, verbose=False, cov_v=cov_v, cov_omega=cov_omega, cov_meas=cov_meas)
print_results(Omega, R, p, v, x, rank, S)
visualize_results(N, K, x, v, R, p, Omega, log=False)

time.sleep(0.1)  # Let things finish printing
gap = compute_relaxation_gap(y_bar, x, v, R, p, Omega, 
                                    x_gt, v_gt, R_gt, p_gt, Omega_gt, 
                                    cov_v, cov_omega, cov_meas)
print(f"Relaxation gap: {gap}")

mean_errors = compute_mean_errors(y_bar, x, v, R, p, Omega, 
                                         x_gt, v_gt, R_gt, p_gt, Omega_gt)
print(f"mean_errors: {mean_errors}")


# Generate a new `testX.py` file to save this test case
# generate_test_file("test_data/benchmark_SPIRAL_N=8_K=8.py", y_bar, x_gt, v_gt, p_gt, R_gt, Omega_gt)
