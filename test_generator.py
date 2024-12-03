import numpy as np
from scipy.spatial.transform import Rotation   
from certifiable_solver import certifiable_solver
from solver_utils import *
from visualization_utils import visualize_results, visualize_results_3D_simple

true_lin_vel = np.array([1, 0, 0.1])
true_rpy_vel = np.array([0, 0, 7])

num_landmarks = 10
num_timesteps = 6

# Define ranges for each dimension
x_range = (0, 10)
y_range = (-2, 5)
z_range = (-3, 3)

x_coords = np.random.uniform(x_range[0], x_range[1], size=num_landmarks)
y_coords = np.random.uniform(y_range[0], y_range[1], size=num_landmarks)
z_coords = np.random.uniform(z_range[0], z_range[1], size=num_landmarks)
true_landmarks = np.stack((x_coords, y_coords, z_coords), axis=1)

true_ang_vel = Rotation.from_euler("xyz", true_rpy_vel, degrees=True).as_matrix()

true_lin_pos, true_ang_pos = generate_ground_truth(num_timesteps, true_lin_vel, true_ang_vel)
print_ground_truth(true_ang_vel, true_ang_pos, true_landmarks, true_lin_vel, true_lin_pos)

# measurements = generate_measurements(true_lin_pos, true_ang_pos, true_landmarks, noise=0.1, dropout=0.1)
# calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S = certifiable_solver(measurements)
# print_results(calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S)
# # visualize_results_3D_simple(calc_landmarks, calc_lin_pos)
# visualize_results(num_timesteps, num_landmarks, calc_lin_pos, calc_lin_vel, calc_ang_pos, calc_landmarks, calc_ang_vel, log=False)

# Generate a new `testX.py` file to save this test case
measurements = generate_measurements(true_lin_pos, true_ang_pos, true_landmarks, noise=0, dropout=0)
generate_test_file("test_data/test7.py", measurements, true_lin_pos, true_lin_vel, true_landmarks, true_ang_pos, true_ang_vel)
