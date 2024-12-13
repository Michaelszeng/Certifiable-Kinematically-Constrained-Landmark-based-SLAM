import numpy as np
np.set_printoptions(edgeitems=30, linewidth=270, precision=4, suppress=True)
from scipy.spatial.transform import Rotation   
import os
import sys
import importlib
import time
from enum import Enum

from solver_utils import *
from visualization_utils import visualize_results
from evaluation import compute_relaxation_gap, compute_mean_errors

class Solver(Enum):
    cvxpy = 1
    drake1 = 2
    drake2 = 3
    nonlinear = 4

################################################################################
##### USER-DEFINED PARAMETERS ##################################################
################################################################################
SOLVER = Solver.nonlinear  # Select a solver from the Solver enums

if SOLVER == Solver.nonlinear:
    def initial_guesses():
        # If you want to see the nonlinear solver converge to a good solution, you'll need to give it a good initial guess.
        # Otherwise, leave these values as "None" and see why a certifiably correct solver is important, especially when good initial guesses are not known.
        # Here, we simply add some noise to the ground truth solution.
        t_guess = t_gt + np.random.normal(loc=0, scale=0.1, size=t_gt.shape)
        R_guess = R_gt + np.random.normal(loc=0, scale=0.1, size=R_gt.shape)
        v_guess = v_gt + np.random.normal(loc=0, scale=0.1, size=v_gt.shape)
        Omega_guess = Omega_gt + np.random.normal(loc=0, scale=0.1, size=Omega_gt.shape)
        p_guess = p_gt + np.random.normal(loc=0, scale=0.1, size=p_gt.shape)
        return t_guess, R_guess, v_guess, Omega_guess, p_guess

PRESET_TEST = None  # i.e. "test1"; Set to None to generate a test case with random landmark locations, or set to a specific test file to run that test case

MEASUREMENT_NOISE = 0.01         # Standard deviation of Gaussian noise added to measurements
MEASUREMENT_DROPOUT = 0.01       # Probability of measurements being dropped (to simulate occlusions or object detection failures)

# Manually define the ground truth trajectory
if PRESET_TEST is None:
    # Example: Spiral trajectory
    K = 4  # Number of landmarks
    N = 4  # Number of timesteps
    d = 3  # Number of dimensions of space
    velocity_trajectory = np.array([[1, 0, 0.5]]*(N-1))
    angular_velocity_trajectory = np.array([[0, 0, 45]]*(N-1))  # RPY angular velocity in degrees

    # Example: Snake trajectory
    # K = 4  # Number of landmarks
    # N = 8  # Number of timesteps
    # d = 3  # Number of dimensions of space
    #v_gt = np.array([[1, 0, 0]]*(N-1))
    #true_rpy_vel = np.array([
    #    [0, 0, 0],
    #    [0, 0, 10],
    #    [0, 0, 20],
    #    [0, 0, 10],
    #    [0, 0, -10],
    #    [0, 0, -20],
    #    [0, 0, -10],
    #])  # RPY angular velocity in degrees
    
    # Velocity, angular velocity, and measurement covariances
    cov_v = 1
    cov_omega = 1
    cov_meas = 1

    # Define ranges for each dimension
    x_range = (-10, 10)
    y_range = (-10, 10)
    z_range = (-10, 10)
################################################################################
################################################################################
################################################################################

# Generate Test Case Data
if PRESET_TEST is None:
    x_coords = np.random.uniform(x_range[0], x_range[1], size=K)
    y_coords = np.random.uniform(y_range[0], y_range[1], size=K)
    z_coords = np.random.uniform(z_range[0], z_range[1], size=K)
    p_gt = np.stack((x_coords, y_coords, z_coords), axis=1)

    Omega_gt = Rotation.from_euler("xyz", angular_velocity_trajectory, degrees=True).as_matrix()

    v_gt = velocity_trajectory
    t_gt, R_gt = generate_ground_truth(N, v_gt, Omega_gt)
    print_ground_truth(Omega_gt, R_gt, p_gt, v_gt, t_gt)

    y_bar = generate_measurements(t_gt, R_gt, p_gt, noise=MEASUREMENT_NOISE, dropout=MEASUREMENT_DROPOUT)
else:
    # Dynamically import the specified test file
    current_folder = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(current_folder, "test_data")
    sys.path.append(test_data_path)
    test_module = importlib.import_module(PRESET_TEST)
    # Import all variables and functions into the current namespace
    globals().update(vars(test_module))


# Import the correct solver
if SOLVER == Solver.cvxpy:
    from solver_cvxpy import solver
    d = 3  # CVXPY solver currently is programmed for 3D postions/rotations only
elif SOLVER == Solver.drake1:
    from solver_drake1 import solver
elif SOLVER == Solver.drake2:
    from solver_drake2 import solver
elif SOLVER == Solver.nonlinear:    
    from solver_nonlinear import solver

# Give initial guesses to the nonlinear solver
if SOLVER == Solver.nonlinear:
    t_guess, R_guess, v_guess, Omega_guess, p_guess = initial_guesses()
    Omega, R, p, v, t, rank, S = solver(y_bar, N, K, d, verbose=False, cov_v=cov_v, cov_omega=cov_omega, cov_meas=cov_meas, t_guess=t_guess, R_guess=R_guess, v_guess=v_guess, Omega_guess=Omega_guess, p_guess=p_guess)
else:
    Omega, R, p, v, t, rank, S = solver(y_bar, N, K, d, verbose=False, cov_v=cov_v, cov_omega=cov_omega, cov_meas=cov_meas)
    
# Process Results
print_results(Omega, R, p, v, t, rank, S)
visualize_results(N, K, t, v, R, p, Omega, log=False)

time.sleep(0.1)  # Let things finish printing
gap = compute_relaxation_gap(y_bar, t, v, R, p, Omega, 
                                    t_gt, v_gt, R_gt, p_gt, Omega_gt, 
                                    cov_v, cov_omega, cov_meas)
print(f"Relaxation gap: {gap}")

mean_errors = compute_mean_errors(y_bar, t, v, R, p, Omega, 
                                         t_gt, v_gt, R_gt, p_gt, Omega_gt)
print(f"mean_errors: {mean_errors}")

# Save test case data if user chooses
if PRESET_TEST is None:
    test_file = input("\nEnter the name of the file to save this test case, i.e. 'test1' (leave empty to not save): ")
    if test_file:
        generate_test_file(f"test_data/{test_file}.py", y_bar, t_gt, v_gt, p_gt, R_gt, Omega_gt)