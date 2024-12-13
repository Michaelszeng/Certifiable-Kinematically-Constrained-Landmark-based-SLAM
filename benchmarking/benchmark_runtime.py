import sys
import os
curr = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(curr)
sys.path.append(parent)

import numpy as np
import time
from scipy.spatial.transform import Rotation
from solver_cvxpy import certifiable_solver
from solver_utils import *

trials = [
#    {
#        "name": "line_small",
#        "true_lin_vel": np.array([1, 0, 0]),
#        "true_rpy_vel": np.array([0, 0, 0]),
#    },
    {
        "name": "spiral_small",
        "true_lin_vel": np.array([1, 0, 0.5]),
        "true_rpy_vel": np.array([0, 0, 45]),
    },
]

samples_per_noise = 2
num_landmarks = [4, 8, 12, 16, 20, 24]
num_timesteps = [3, 4, 5, 6, 7, 8]

for trial in trials:
    name = trial["name"]
    true_lin_vel = trial["true_lin_vel"]
    true_rpy_vel = trial["true_rpy_vel"]

    average_times_lm = []
    average_times_t = []

    for num_lm in num_landmarks:
        times = []
        print(f"Number of landmarks: {num_lm}\n")
        for _ in range(samples_per_noise):
            true_landmarks = np.random.uniform(-10, 10, size=(num_lm, 3))
            true_ang_vel = Rotation.from_euler("xyz", true_rpy_vel, degrees=True).as_matrix()

            true_lin_pos, true_ang_pos = generate_ground_truth(4, true_lin_vel, true_ang_vel)
            print_ground_truth(true_ang_vel, true_ang_pos, true_landmarks, true_lin_vel, true_lin_pos)

            measurements = generate_measurements(true_lin_pos, true_ang_pos, true_landmarks, noise=0.0)
            
            start_time = time.time()
            calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S = certifiable_solver(measurements)
            elapsed_time = time.time() - start_time

            print_results(calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S)
            print(f"Elapsed time: {elapsed_time}\n")

            times.append(elapsed_time)
        
        average_times_lm.append(np.mean(times))

    for num_t in num_timesteps:
        times = []
        print(f"Number of timesteps: {num_t}\n")
        for _ in range(samples_per_noise):
            true_landmarks = np.random.uniform(-10, 10, size=(4, 3))
            true_ang_vel = Rotation.from_euler("xyz", true_rpy_vel, degrees=True).as_matrix()

            true_lin_pos, true_ang_pos = generate_ground_truth(num_t, true_lin_vel, true_ang_vel)
            print_ground_truth(true_ang_vel, true_ang_pos, true_landmarks, true_lin_vel, true_lin_pos)

            measurements = generate_measurements(true_lin_pos, true_ang_pos, true_landmarks, noise=0.0)
            
            start_time = time.time()
            calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S = certifiable_solver(measurements)
            elapsed_time = time.time() - start_time

            print_results(calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S)
            print(f"Elapsed time: {elapsed_time}\n")

            times.append(elapsed_time)
        
        average_times_t.append(np.mean(times))

    average_times_lm = np.array(average_times_lm)
    average_times_t = np.array(average_times_t)

    np.save(f"output_files/{name}_times_lm.npy", average_times_lm)
    np.save(f"output_files/{name}_times_t.npy", average_times_t)

