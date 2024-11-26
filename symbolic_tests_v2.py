from pydrake.all import (
    MathematicalProgram,
    Solve,
)

import cvxpy as cp
import numpy as np
import sys
import os

from visualization_utils import visualize_results

current_folder = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(current_folder, "test_data")
sys.path.append(test_data_path)
from test2 import *

np.set_printoptions(edgeitems=30, linewidth=200, precision=4, suppress=True)

# Full State:
#      d*N d*N d*K, d*d*N, d*d*N
# x = [ t,  v,  p,    R,   Omega]

prog = MathematicalProgram()

# Variable Definitions
# NOTE: DEFINE THESE IN THE ORDER THEY APPEAR IN OUR FULL STATE REPRESENTATION
t = [prog.NewContinuousVariables(d, f"t_{i}") for i in range(N)]                # Positions t_i
v = [prog.NewContinuousVariables(d, f"v_{i}") for i in range(N)]                # Velocities v_i
p = [prog.NewContinuousVariables(d, f"p_{k}") for k in range(K)]                # Landmark positions p_ks
R = [prog.NewContinuousVariables(d, d, f"R_{i}") for i in range(N)]             # Rotations R_i
Omega = [prog.NewContinuousVariables(d, d, f"Omega_{i}") for i in range(N)]     # Angular velocities Ω_i


# Constraint Definitions
Q_constraint = np.zeros((d*N + d*N + d*K + d*d*N + d*d*N, d*N + d*N + d*K + d*d*N + d*d*N))
b_constraint = np.zeros(d*N + d*N + d*K + d*d*N + d*d*N)

# 1. Constant Twist Constraints
for i in range(N - 1):
    # Position update: t_{i+1} = t_i + R_i @ v_i
    for dim in range(d):
        # Compute the dim'th element of the matrix vector product R_i @ v_i
        rotation_times_velocity = sum(R[i][dim, k] * v[i][k] for k in range(d))
        constraint_binding = prog.AddConstraint(t[i + 1][dim] == t[i][dim] + rotation_times_velocity)
        
        constraint = constraint_binding.evaluator()
        # Each dimension of constraint.Q() represents: [t_i[dim], t_{i+1}[dim], v_i[0], v_i[1], v_i[2], R_i[0,0], R_i[0,1], R_i[0,2]]
        # print(constraint.Q())
        # print(constraint.b())
        Q_constraint[d*N + d*i : d*N + d*(i+1), d*N + d*N + d*K + d*d*i + d*dim : d*N + d*N + d*K + d*d*i + d*(dim+1)] += constraint.Q()[2:5, 5:8]  # v,r
        Q_constraint[d*N + d*N + d*K + d*d*i + d*dim : d*N + d*N + d*K + d*d*i + d*(dim+1), d*N + d*i : d*N + d*(i+1)] += constraint.Q()[5:8, 2:5]  # r,v
        b_constraint[d*i + dim : d*i + dim + 1] += constraint.b()[0]  # t_i[dim]
        b_constraint[d*(i+1) + dim : d*(i+1) + dim + 1] += constraint.b()[1]  # t_{i+1}[dim]

    
    # Rotation update: R_{i+1} = R_i @ Omega_i
    for row in range(d):
        for col in range(d):
            # Compute the (row, col) element of the matrix multiplication R_i @ Omega_i
            rotation_element = 0
            for j in range(d):
                rotation_element += R[i][row, j] * Omega[i][j, col]
            constraint_binding = prog.AddConstraint(R[i + 1][row, col] == rotation_element)
            
            constraint = constraint_binding.evaluator()
            print(constraint.Q())
            print(constraint.b())
            

# 2. SO(3) Constraints: R_i^T @ R_i == I_d
for i in range(N):
    for row in range(d):
        for col in range(d):
            if row == col:
                # Diagonal entries
                constraint_binding_R = prog.AddConstraint(R[i][:, row].dot(R[i][:, col]) == 1)
                constraint_binding_Omega = prog.AddConstraint(Omega[i][:, row].dot(Omega[i][:, col]) == 1)
            else:
                # Off-diagonal entries
                constraint_binding_R = prog.AddConstraint(R[i][:, row].dot(R[i][:, col]) == 0)
                constraint_binding_Omega = prog.AddConstraint(Omega[i][:, row].dot(Omega[i][:, col]) == 0)
                
            constraint_R = constraint_binding_R.evaluator()
            # print(np.shape(constraint_R.Q()))
            # print(np.shape(constraint_R.b()))
            
            constraint_Omega = constraint_binding_Omega.evaluator()
            # print(np.shape(constraint_Omega.Q()))
            # print(np.shape(constraint_Omega.b()))


# Objective Function
Q_cost = np.zeros((d*N + d*N + d*K + d*d*N + d*d*N, d*N + d*N + d*K + d*d*N + d*d*N))
print(f"Q_cost: {np.shape(Q_cost)}")

# 1. Landmark Residuals
for k in range(K):
    for j, y_bar_kj in y_bar[k].items():
        # R[j] @ y_bar[k][j]
        Rj_y = [sum(R[j][row, m] * y_bar_kj[m] for m in range(d)) for row in range(d)]
        
        # (p[k] - t[j])
        t_minus_p = [p[k][dim] - t[j][dim] for dim in range(d)]
        
        # Residual: R[j] @ y_bar[k][j] - (t[j] - p[k])
        residual = [Rj_y[row] - t_minus_p[row] for row in range(d)]
        
        # Quadratic form: residual^T * Sigma_p * residual
        quad_form = 0.0
        for r in range(d):
            for c in range(d):
                quad_form += residual[r] * Sigma_p[r, c] * residual[c]
        
        cost_binding = prog.AddCost(quad_form)
        cost = cost_binding.evaluator()
        # print(cost.Q())
        # print(np.kron(np.array(y_bar_kj).reshape(3,1).T, Sigma_p))  # t,r = -p,r
        # print(np.kron(np.array(y_bar_kj).reshape(3,1) @ np.array(y_bar_kj).reshape(3,1).T, Sigma_p))  # r,r
        Q_cost[d*j : d*(j+1), d*j : d*(j+1)] += cost.Q()[0:3,0:3]  # t,t
        Q_cost[d*j : d*(j+1), 2*d*N + d*k : 2*d*N + d*(k+1)] += cost.Q()[0:3,3:6]  # t,p
        Q_cost[2*d*N + d*k : 2*d*N + d*(k+1), d*j : d*(j+1)] += cost.Q()[3:6,0:3]  # p,t
        Q_cost[2*d*N + d*k : 2*d*N + d*(k+1), 2*d*N + d*k : 2*d*N + d*(k+1)] += cost.Q()[3:6,3:6]  # p,p
        Q_cost[d*j : d*(j+1), 2*d*N + d*K + d*d*j : 2*d*N + d*K + d*d*(j+1)] += cost.Q()[0:3,6:15]  # t,r
        Q_cost[2*d*N + d*K + d*d*j : 2*d*N + d*K + d*d*(j+1), d*j : d*(j+1)] += cost.Q()[6:15,0:3]  # r,t
        Q_cost[2*d*N + d*K + d*d*j : 2*d*N + d*K + d*d*(j+1), 2*d*N + d*K + d*d*j : 2*d*N + d*K + d*d*(j+1)] += cost.Q()[6:15,6:15]  #r,r
        
        
# 2. Velocity Differences
for i in range(N - 1):
    # v_{i+1} - v_i
    v_diff = [v[i + 1][dim] - v[i][dim] for dim in range(d)]
    
    # Quadratic form: v_diff^T * Sigma_v * v_diff
    quad_form_v = 0.0
    for r in range(d):
        for c in range(d):
            quad_form_v += v_diff[r] * Sigma_v[r, c] * v_diff[c]
    
    cost_binding = prog.AddCost(quad_form_v)

    cost = cost_binding.evaluator()
    # print(cost.Q())
    Q_cost[d*N + d*i : d*N + d*(i+2), d*N + d*i : d*N + d*(i+2)] += cost.Q()

# 3. Angular Velocity Differences
for i in range(N - 1):
    # Omega_{i+1} - Omega_i, flattened
    Omega_diff = [Omega[i + 1][k, l] - Omega[i][k, l] for k in range(d) for l in range(d)]
    
    # Quadratic form: Omega_diff^T * Sigma_omega * Omega_diff
    quad_form_omega = 0.0
    for r in range(d**2):
        for c in range(d**2):
            quad_form_omega += Omega_diff[r] * Sigma_omega[r, c] * Omega_diff[c]
    
    cost_binding = prog.AddCost(quad_form_omega)
    
    cost = cost_binding.evaluator()
    # print(cost.Q())
    Q_cost[2*d*N + d*K + d*d*N + d*d*i : 2*d*N + d*K + d*d*N + d*d*(i+2), 2*d*N + d*K + d*d*N + d*d*i : 2*d*N + d*K + d*d*N + d*d*(i+2)] += cost.Q()

# print(Q_cost)

# Set initial guesses and Solve
for i in range(N):
    prog.SetInitialGuess(t[i], t_guess[i])
    prog.SetInitialGuess(v[i], t_guess[i])
    prog.SetInitialGuess(R[i], R_guess[i])
    prog.SetInitialGuess(Omega[i], Omega_guess[i])
for k in range(K):
    prog.SetInitialGuess(p[k], p_guess[k])
    
result = Solve(prog)

if result.is_success():
    t_sol = []
    v_sol = []
    R_sol = []
    Omega_sol = []
    p_sol = []
    for i in range(N):
        t_sol.append(result.GetSolution(t[i]))
        v_sol.append(result.GetSolution(v[i]))
        R_sol.append(result.GetSolution(R[i]))
        Omega_sol.append(result.GetSolution(Omega[i]))
    for k in range(K):
        p_sol.append(result.GetSolution(p[k]))
    pass
else:
    print("solve failed.")
    

visualize_results(N, K, t_sol, v_sol, R_sol, p_sol)