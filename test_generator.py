import numpy as np
import time
from scipy.spatial.transform import Rotation   
from certifiable_solver import certifiable_solver
from solver_utils import *
from visualization_utils import visualize_results
from evaluation import compute_relaxation_gap, compute_mean_errors

true_lin_vel = np.array([1, 0, 0])
true_rpy_vel = np.array([0, 0, 0])

num_landmarks = 10
num_timesteps = 10

# Velocity, angular velocity, and measurement covariances
cov_v = 1
cov_omega = 1
cov_meas = 1

# Define ranges for each dimension
x_range = (-10, 10)
y_range = (-10, 10)
z_range = (-10, 10)

x_coords = np.random.uniform(x_range[0], x_range[1], size=num_landmarks)
y_coords = np.random.uniform(y_range[0], y_range[1], size=num_landmarks)
z_coords = np.random.uniform(z_range[0], z_range[1], size=num_landmarks)
true_landmarks = np.stack((x_coords, y_coords, z_coords), axis=1)

true_ang_vel = Rotation.from_euler("xyz", true_rpy_vel, degrees=True).as_matrix()

true_lin_pos, true_ang_pos = generate_ground_truth(num_timesteps, true_lin_vel, true_ang_vel)
print_ground_truth(true_ang_vel, true_ang_pos, true_landmarks, true_lin_vel, true_lin_pos)

measurements = generate_measurements(true_lin_pos, true_ang_pos, true_landmarks, noise=0.0, dropout=0.0)


# calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S = certifiable_solver(
#         measurements, verbose=False, cov_v=cov_v, cov_omega=cov_omega, cov_meas=cov_meas)
# print_results(calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S)
# visualize_results(num_timesteps, num_landmarks, calc_lin_pos, calc_lin_vel, calc_ang_pos, calc_landmarks, calc_ang_vel, log=False)

# time.sleep(0.1)  # Let things finish printing
# gap = compute_relaxation_gap(measurements, calc_lin_pos, calc_lin_vel, calc_ang_pos, calc_landmarks, calc_ang_vel, 
#                                            true_lin_pos, true_lin_vel, true_ang_pos, true_landmarks, true_ang_vel, 
#                                            cov_v, cov_omega, cov_meas)
# print(f"Relaxation gap: {gap}")


# mean_errors = compute_mean_errors(measurements, calc_lin_pos, calc_lin_vel, calc_ang_pos, calc_landmarks, calc_ang_vel, 
#                                                 true_lin_pos, true_lin_vel, true_ang_pos, true_landmarks, true_ang_vel)
# print(f"mean_errors: {mean_errors}")


# Generate a new `testX.py` file to save this test case
generate_test_file("test_data/benchmark_LINE_N=10_K=10.py", measurements, true_lin_pos, true_lin_vel, true_landmarks, true_ang_pos, true_ang_vel)