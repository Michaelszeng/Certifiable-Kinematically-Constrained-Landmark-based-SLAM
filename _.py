"""
area to run dumb tests
"""

from pydrake.all import (
    MathematicalProgram,
    Solve,
)
import os
import sys

current_folder = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(current_folder, "test_data")
sys.path.append(test_data_path)
from test1 import *


prog = MathematicalProgram()

t = [prog.NewContinuousVariables(d, f"t_{i}") for i in range(N)]                # Positions t_i
R = [prog.NewContinuousVariables(d, d, f"R_{i}") for i in range(N)]             # Rotations R_i
v = [prog.NewContinuousVariables(d, f"v_{i}") for i in range(N)]                # Velocities v_i
Omega = [prog.NewContinuousVariables(d, d, f"Omega_{i}") for i in range(N)]     # Angular velocities Ω_i
p = [prog.NewContinuousVariables(d, f"p_{k}") for k in range(K)]                # Landmark positions p_ks



# 1. Constant Twist Constraints
for i in range(N - 1):
    # Position update: t_{i+1} = t_i + v_i
    for dim in range(d):
        prog.AddLinearEqualityConstraint(t[i + 1][dim] == t[i][dim] + v[i][dim])
    
    # Rotation update: R_{i+1} = R_i @ Omega_i
    for row in range(d):
        for col in range(d):
            # Compute the (row, col) element of the matrix multiplication R_i @ Omega_i
            rotation_element = 0
            for i in range(d):
                rotation_element += R[i][row, i] * Omega[i][i, col]
            prog.AddConstraint(R[i + 1][row, col] == rotation_element)



# 2. SO(3) Constraints: R_i^T @ R_i == I_d
for i in range(N):
    for row in range(d):
        for col in range(d):
            if row == col:
                # Diagonal entries
                prog.AddConstraint(R[i][:, row].dot(R[i][:, col]) == 1)
                prog.AddConstraint(Omega[i][:, row].dot(Omega[i][:, col]) == 1)
            else:
                # Off-diagonal entries
                prog.AddConstraint(R[i][:, row].dot(R[i][:, col]) == 0)
                prog.AddConstraint(Omega[i][:, row].dot(Omega[i][:, col]) == 0)



for k in range(K):
    for j, y_bar_kj in y_bar[k].items():
        # R[j] @ y_bar[k][j]
        Rj_y = [sum(R[j][row, m] * y_bar_kj[m] for m in range(d)) for row in range(d)]
        
        # (t[j] - p[k])
        t_minus_p = [t[j][dim] - p[k][dim] for dim in range(d)]
        
        # Residual: R[j] @ y_bar[k][j] - (t[j] - p[k])
        residual = [Rj_y[row] - t_minus_p[row] for row in range(d)]
        
        # Quadratic form: residual^T * Sigma_p * residual
        quad_form = 0.0
        for r in range(d):
            for c in range(d):
                quad_form += residual[r] * Sigma_p[r, c] * residual[c]
        
        prog.AddCost(quad_form)

print(prog.quadratic_costs())
print(len(prog.quadratic_costs()))



for i in range(N):
    prog.SetInitialGuess(t[i], t_guess[i])
    prog.SetInitialGuess(v[i], t_guess[i])
    prog.SetInitialGuess(R[i], R_guess[i])
    prog.SetInitialGuess(Omega[i], Omega_guess[i])
for k in range(K):
    prog.SetInitialGuess(p[k], p_guess[k])


result = Solve(prog)

if result.is_success():
    # for i in range(N):
    #     print(result.GetSolution(f"t{i}: {t[i]}"))
    #     print(result.GetSolution(f"v{i}: {v[i]}"))
    #     print(result.GetSolution(f"R{i}: {R[i]}"))
    #     print(result.GetSolution(f"Omega{i}: {Omega[i]}"))
    # for k in range(K):
    #     print(result.GetSolution(f"p{k}: {p[k]}"))
    pass
else:
    print("fail")
