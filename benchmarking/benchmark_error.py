import sys
import os
curr = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(curr)
sys.path.append(parent)

import numpy as np
from scipy.spatial.transform import Rotation
from certifiable_solver import certifiable_solver
from solver_utils import *
from evaluation import compute_relaxation_gap, compute_mean_errors

trials = [
    {
        "name": "line_small",
        "true_lin_vel": np.array([1, 0, 0]),
        "true_rpy_vel": np.array([0, 0, 0]),
        "num_landmarks": 4,
        "num_timesteps": 4,
    },
    {
        "name": "spiral_small",
        "true_lin_vel": np.array([1, 0, 0.5]),
        "true_rpy_vel": np.array([0, 0, 45]),
        "num_landmarks": 4,
        "num_timesteps": 4,
    },
    {
        "name": "curve",
        "true_lin_vel": np.array([1, 0, 0]),
        "true_rpy_vel": np.array([
            [0, 0, 0],
            [0, 0, 10],
            [0, 0, 20],
            [0, 0, 10],
            [0, 0, -10],
            [0, 0, -20],
            [0, 0, -10],
        ]),
        "num_landmarks": 4,
        "num_timesteps": 8,
    },
]

samples_per_noise = 10
noise_levels = [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0]

for trial in trials:
    name = trial["name"]
    true_lin_vel = trial["true_lin_vel"]
    true_rpy_vel = trial["true_rpy_vel"]
    num_landmarks = trial["num_landmarks"]
    num_timesteps = trial["num_timesteps"]

    averages = {
        "gap": [],
        "t": [],
        "v": [],
        "R": [],
        "p": [],
        "Omega": [],
    }

    for noise in noise_levels:
        curr_averages = {
            "gap": [],
            "t": [],
            "v": [],
            "R": [],
            "p": [],
            "Omega": [],
        }
        print(f"Noise: {noise}\n")
        for _ in range(samples_per_noise):
            true_landmarks = np.random.uniform(-10, 10, size=(num_landmarks, 3))
            true_ang_vel = Rotation.from_euler("xyz", true_rpy_vel, degrees=True).as_matrix()

            true_lin_pos, true_ang_pos = generate_ground_truth(num_timesteps, true_lin_vel, true_ang_vel)
            print_ground_truth(true_ang_vel, true_ang_pos, true_landmarks, true_lin_vel, true_lin_pos)

            measurements = generate_measurements(true_lin_pos, true_ang_pos, true_landmarks, noise=noise)
            
            calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S = certifiable_solver(measurements)
            print_results(calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S)

            gap = compute_relaxation_gap(measurements, calc_lin_pos, calc_lin_vel, calc_ang_pos, calc_landmarks, calc_ang_vel, 
                                         true_lin_pos, true_lin_vel, true_ang_pos, true_landmarks, true_ang_vel)
            curr_averages["gap"].append(gap)

            mean_errors = compute_mean_errors(measurements, calc_lin_pos, calc_lin_vel, calc_ang_pos, calc_landmarks, calc_ang_vel, 
                                                            true_lin_pos, true_lin_vel, true_ang_pos, true_landmarks, true_ang_vel)    
            for key, val in mean_errors.items():
                curr_averages[key].append(val)

        for key, val in curr_averages.items():
            averages[key].append(float(np.mean(val)))

    np.save(f"{name}_error.npy", averages)

