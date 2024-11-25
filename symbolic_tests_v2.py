from pydrake.all import (
    MathematicalProgram,
    Solve,
)

import cvxpy as cp
import numpy as np
import sys
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(current_folder, "test_data")
sys.path.append(test_data_path)
from test1 import *

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

# 1. Constant Twist Constraints
for i in range(N - 1):
    # Position update: t_{i+1} = t_i + R_i @ v_i
    for dim in range(d):
        # Compute the dim'th element of the matrix vector product R_i @ v_i
        rotation_times_velocity = sum(R[i][dim, k] * v[i][k] for k in range(d))
        prog.AddConstraint(t[i + 1][dim] == t[i][dim] + rotation_times_velocity)
    
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

print(Q_cost)

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
    


def visualize_results(N, K, t, v, R, p):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    def draw_triangle(ax, position, rotation_matrix, size=0.3, color='blue'):
        """
        Draw a triangle to represent the robot pose.
        Args:
            ax: Matplotlib axis
            position: 2D position of the robot (x, y)
            rotation_matrix: 2x2 rotation matrix
            size: Size of the triangle
            color: Color of the triangle
        """
        # Define a basic triangle pointing up (relative to local frame)
        triangle = np.array([[0, size], [size/2, -size/2], [-size/2, -size/2]])
        # Rotate and translate triangle
        rotated_triangle = (rotation_matrix @ triangle.T).T + position
        polygon = Polygon(rotated_triangle, closed=True, color=color)
        ax.add_patch(polygon)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    # Plot each robot pose as a triangle and velocity as an arrow
    for i in range(N):
        position = np.array([t[i][0], t[i][1]])  # Ignore z-axis
        rotation_matrix = R[i][:2, :2]  # Use the top-left 2x2 of the 3x3 matrix
        v_i_rotated = R[i] @ v[i]
        velocity = np.array([v_i_rotated[0], v_i_rotated[1]])  # Ignore z-axis
        
        # Draw robot pose
        draw_triangle(ax, position, rotation_matrix, size=0.3, color='blue')
        
        # Draw velocity vector
        ax.quiver(
            position[0], position[1],
            velocity[0], velocity[1],
            angles='xy', scale_units='xy', scale=1, color='green'
        )

    # Plot each point p[k] in red
    for k in range(K):
        point = np.array([p[k][0], p[k][1]])  # Ignore z-axis
        ax.plot(point[0], point[1], 'ro')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Visualization of Robot Poses and Points')

    plt.grid()
    plt.show()

# visualize_results(N, K, t_sol, v_sol, R_sol, p_sol)
    
    
# Retrieve Q and b matrices and formulate Standard Form QCQP
for quad_cost in prog.quadratic_costs():
    quad_cost = quad_cost.evaluator()
    # print(np.shape(quad_cost.Q()))
    # print(np.shape(quad_cost.b()))
    # print(np.shape(quad_cost.c()))
    
print("=======================================================================")
    
for quad_constraint in prog.quadratic_constraints():
    quad_constraint = quad_constraint.evaluator()
    # print(np.shape(quad_constraint.Q()))
    # print(np.shape(quad_constraint.b()))
