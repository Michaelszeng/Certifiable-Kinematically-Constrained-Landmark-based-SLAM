import numpy as np
from scipy.spatial.transform import Rotation   
from certifiable_solver import certifiable_solver
from solver_utils import *
from visualization_utils import visualize_results_3D_simple

true_lin_vel = np.array([2, 0, 1])
true_rpy_vel = np.array([30, 0, 0])

num_landmarks = 6
num_timesteps = 5

true_landmarks = np.random.uniform(-20, 20, size=(num_landmarks, 3))
true_ang_vel = Rotation.from_euler("xyz", true_rpy_vel, degrees=True).as_matrix()

true_lin_pos, true_ang_pos = generate_ground_truth(num_timesteps, true_lin_vel, true_ang_vel)
print_ground_truth(true_ang_vel, true_ang_pos, true_landmarks, true_lin_vel, true_lin_pos)

measurements = generate_measurements(true_lin_pos, true_ang_pos, true_landmarks, noise=0.1, dropout=0.1)
calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S = certifiable_solver(measurements)
print_results(calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S)
visualize_results_3D_simple(calc_landmarks, calc_lin_pos)

# Generate a new `testX.py` file to save this test case
# generate_python_file("test_data/test4.py", measurements, true_lin_pos, true_lin_vel, true_landmarks, true_ang_pos, true_ang_vel)