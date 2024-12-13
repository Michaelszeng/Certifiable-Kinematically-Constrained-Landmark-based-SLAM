from pydrake.all import (
    MathematicalProgram,
    Solve,
    SolverOptions,
    MosekSolver,
)

import numpy as np
import pandas as pd 
import sys
import os
import time

from visualization_utils import *

def solver(y_bar, N, K, d, verbose=False, tol=1e-6, cov_v=1, cov_omega=1, cov_meas=1):
    # Expand out covariance matrices
    Sigma_p = np.linalg.inv(cov_meas*np.eye(d))  # Covariance matrix for position
    Sigma_v = np.linalg.inv(cov_v*np.eye(d))  # Covariance matrix for velocity
    Sigma_omega = np.linalg.inv(cov_omega*np.eye(d**2))  # Covariance matrix for angular velocity
    
    prog = MathematicalProgram()

    # Variable Definitions
    # NOTE: DEFINE THESE IN THE ORDER THEY APPEAR IN OUR FULL STATE REPRESENTATION
    t = [prog.NewContinuousVariables(d, f"t_{i}") for i in range(N)]                # Positions t_i
    p = [prog.NewContinuousVariables(d, f"p_{k}") for k in range(K)]                # Landmark positions p_k
    R = [prog.NewContinuousVariables(d, d, f"R_{i}") for i in range(N)]             # Rotations R_i
    Omega = [prog.NewContinuousVariables(d, d, f"Ω_{i}") for i in range(N-1)]     # Angular velocities Ω_i

    v = [prog.NewContinuousVariables(d, f"v_{i}") for i in range(N-1)]                # Velocities v_i

    # 1. Linear Odometry Constraint
    for i in range(N - 1):
        # Position update: R_i^T @ t_{i+1} = R_i^T @ t_i + v_i
        for dim in range(d):
            # Compute the dim'th element of the matrix vector product R_i @ t_{i+1}
            R_t_i_plus_1 = sum(R[i].T[dim, j] * t[i+1][j] for j in range(d))
            # Compute the dim'th element of the matrix vector product R_i @ t_i
            R_t_i = sum(R[i].T[dim, j] * t[i][j] for j in range(d))
            
            prog.AddConstraint(R_t_i_plus_1 == R_t_i + v[i][dim])


    # 2. Rotational Odometry Constraint
    for i in range(N - 2):
        # Rotation update: R_{i+1} = R_i @ Omega_i --> R_i @ Omega_i = R_{i+2} @ Omega_{i+1}^T
        for row in range(d):
            for col in range(d):
                # Compute the (row, col) element of the matrix multiplication R_i @ Omega_i
                left_side = 0
                right_side = 0
                for j in range(d):
                    left_side += R[i][row, j] * Omega[i][j, col]
                    right_side += R[i+2][row, j] * Omega[i].T[j, col]
                prog.AddConstraint(left_side == right_side)

    # 3. SO(3) Constraint on Rotation: R_i^T @ R_i == I_d and R_i @ R_i^T == I_d
    for i in range(N):
        for row in range(d):
            for col in range(d):
                if row == col:
                    # Diagonal entries
                    prog.AddConstraint(R[i].T[row, :].dot(R[i][:, col]) == 1)
                    prog.AddConstraint(R[i][row, :].dot(R[i].T[:, col]) == 1)
                else:
                    # Off-diagonal entries
                    prog.AddConstraint(R[i].T[row, :].dot(R[i][:, col]) == 0)
                    prog.AddConstraint(R[i][row, :].dot(R[i].T[:, col]) == 0)

    # 4. SO(3) Constraint on Angular Velocity: Omega_i^T @ Omega_i == I_d and Omega_i @ Omega_i^T == I_d
    for i in range(N-1):
        for row in range(d):
            for col in range(d):
                if row == col:
                    # Diagonal entries
                    prog.AddConstraint(Omega[i].T[row, :].dot(Omega[i][:, col]) == 1)
                    constraint_binding2 = prog.AddConstraint(Omega[i][row, :].dot(Omega[i].T[:, col]) == 1)
                else:
                    # Off-diagonal entries
                    prog.AddConstraint(Omega[i].T[row, :].dot(Omega[i][:, col]) == 0)
                    prog.AddConstraint(Omega[i][row, :].dot(Omega[i].T[:, col]) == 0)
                
    # 5. Initial identity rotation:
    for row in range(d):
        for col in range(d):
            # Identity matrix has the following property:
            # Product of any 2 diagonal elements == 1
            prog.AddConstraint(R[0][row,row]*R[0][col,col] == 1)
            
            if row != col:
                # Off-diagonal entries
                prog.AddConstraint(R[0][row,col]*R[0][col,row] == 0)

    # 6: Initial 0 translation:
    for dim in range(d):
        prog.AddConstraint(t[0][dim] * t[0][dim] == 0)


    # Cost Function
    # Cost is of the form: 1/2 x^T Q_cost x
    Q_cost = np.zeros((prog.num_vars() - d*(N-1), prog.num_vars() - d*(N-1)))
    P_cost = np.zeros((d*(N-1), d*(N-1)))

    # 1. Landmark Residuals
    for k in range(K):
        for j, y_bar_kj in y_bar[k].items():
            # R[j] @ y_bar[k][j]
            Rj_y = [sum(R[j][row, m] * y_bar_kj[m] for m in range(d)) for row in range(d)]
            
            # (p[k] - t[j])
            p_minus_t = [p[k][dim] - t[j][dim] for dim in range(d)]
            
            # Residual: R[j] @ y_bar[k][j] - (p[k] - t[j])
            residual = [Rj_y[row] - p_minus_t[row] for row in range(d)]
            
            # Quadratic form: residual^T * Sigma_p * residual
            quad_form = 0.0
            for r in range(d):
                for c in range(d):
                    quad_form += residual[r] * Sigma_p[r, c] * residual[c]
            
            prog.AddCost(quad_form)

    # 2. Velocity Differences
    for i in range(N - 2):
        # v_{i+1} - v_i
        v_diff = [v[i + 1][dim] - v[i][dim] for dim in range(d)]
        
        # Quadratic form: v_diff^T * Sigma_v * v_diff
        quad_form_v = 0.0
        for r in range(d):
            for c in range(d):
                quad_form_v += v_diff[r] * Sigma_v[r, c] * v_diff[c]
        
        prog.AddCost(quad_form_v)

    # 3. Angular Velocity Differences
    for i in range(N - 2):
        # Omega_{i+1} - Omega_i, flattened
        Omega_diff = [Omega[i + 1][j, l] - Omega[i][j, l] for j in range(d) for l in range(d)]
        
        # Quadratic form: Omega_diff^T * Sigma_omega * Omega_diff
        quad_form_omega = 0.0
        for r in range(d**2):
            for c in range(d**2):
                quad_form_omega += Omega_diff[r] * Sigma_omega[r, c] * Omega_diff[c]
        
        prog.AddCost(quad_form_omega)

    # Set initial guesses and Solve
    for i in range(N):
        prog.SetInitialGuess(t[i], t_guess[i])
        prog.SetInitialGuess(R[i], R_guess[i])
    for i in range(N-1):
        prog.SetInitialGuess(v[i], v_guess[i])
        prog.SetInitialGuess(Omega[i], Omega_guess[i])
    for k in range(K):
        prog.SetInitialGuess(p[k], p_guess[k])
        
    print("Beginning Non-convex Solve.")
    start = time.time()
    result = Solve(prog)
    print(f"Non-convex Solve Time: {time.time() - start}")
    print(f"Solved using: {result.get_solver_id().name()}")

    if result.is_success():
        t_sol = []
        v_sol = []
        R_sol = []
        Omega_sol = []
        p_sol = []
        for i in range(N):
            t_sol.append(result.GetSolution(t[i]))
            R_sol.append(result.GetSolution(R[i]))
        for i in range(N-1):
            v_sol.append(result.GetSolution(v[i]))
            Omega_sol.append(result.GetSolution(Omega[i]))
        for k in range(K):
            p_sol.append(result.GetSolution(p[k]))
        
        return Omega_sol, R_sol, p_sol, v_sol, t_sol, "N/A", "N/A"
        
    else:
        print("solve failed.")