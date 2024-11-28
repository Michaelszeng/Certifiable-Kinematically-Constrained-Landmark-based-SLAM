from pydrake.all import (
    MathematicalProgram,
    Solve,
    SolverOptions,
    MosekSolver,
)

import cvxpy as cp
import numpy as np
import sys
import os
import time

from visualization_utils import visualize_results

current_folder = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(current_folder, "test_data")
sys.path.append(test_data_path)
from test0 import *

np.set_printoptions(edgeitems=30, linewidth=270, precision=4, suppress=True)

# Full State:
#      d*N d*N d*K, d*d*N, d*d*N
# x = [ t,  v,  p,    R,   Omega]


################################################################################
##### NON-CONVEX PROGRAM
################################################################################

prog = MathematicalProgram()

# Variable Definitions
# NOTE: DEFINE THESE IN THE ORDER THEY APPEAR IN OUR FULL STATE REPRESENTATION
t = [prog.NewContinuousVariables(d, f"t_{i}") for i in range(N)]                # Positions t_i
v = [prog.NewContinuousVariables(d, f"v_{i}") for i in range(N)]                # Velocities v_i
p = [prog.NewContinuousVariables(d, f"p_{k}") for k in range(K)]                # Landmark positions p_ks
R = [prog.NewContinuousVariables(d, d, f"R_{i}") for i in range(N)]             # Rotations R_i
Omega = [prog.NewContinuousVariables(d, d, f"Omega_{i}") for i in range(N)]     # Angular velocities Ω_i


def add_constraint_to_qcqp(constraint_binding):
    """
    Helper function to format a generic (quadratic) constraint into QCQP form by
    adding to the `Q_constraint` and `b_constraint` arrays.
    
    TODO: build compatibility with linear costs as well to make this work with
    more general formulations that might have linear constraints.
    
    Args:
        constraint_binding: Binding<Constraint> object containing binding for the added constraint.
        
    Returns:
        None; augments `Q_constraint` and `b_constraint` directly.
    """
    Q_constraint = np.zeros((prog.num_vars(), prog.num_vars()))
    b_constraint = np.zeros(prog.num_vars())
    c_constraint = 0

    constraint = constraint_binding.evaluator()
    constraint_vars = constraint_binding.variables()
        
    for j, v1 in enumerate(constraint_vars):
        v1_idx = prog.FindDecisionVariableIndex(v1)

        for l, v2 in enumerate(constraint_vars):
            v2_idx = prog.FindDecisionVariableIndex(v2)

            Q_constraint[v1_idx, v2_idx] += constraint.Q()[j, l]
        
        b_constraint[v1_idx] = constraint.b()[j]
        
    assert constraint.lower_bound() == constraint.upper_bound()
    c_constraint = -constraint.lower_bound()
        
    Q_constraints.append(Q_constraint)
    b_constraints.append(b_constraint)
    c_constraints.append(c_constraint)
        
    

def add_cost_to_qcqp(cost_binding):
    """
    Helper function to format a generic (quadratic) cost into QCQP form by
    adding to the `Q_cost` matrix.
    
    Note that it is assumed the optimization admits a least squares formulation,
    so there are on linear terms in the cost.
    
    Args:
        cost_binding: Binding<Cost> object containing binding for the added cost.
        
    Returns:
        None; augments `Q_cost` and `b_cost` directly.
    """
    cost = cost_binding.evaluator()
    cost_vars = cost_binding.variables()
        
    for j, v1 in enumerate(cost_vars):
        v1_idx = prog.FindDecisionVariableIndex(v1)

        for l, v2 in enumerate(cost_vars):
            v2_idx = prog.FindDecisionVariableIndex(v2)

            Q_cost[v1_idx, v2_idx] += cost.Q()[j, l]
    

# Constraint Definitions
# Each constraint is of the form: x^T Q_constraints[i] x + b_constraints[i]^T x + c_constraints[i] = 0
Q_constraints = []
b_constraints = []
c_constraints = []

# 1. Constant Twist Constraints
for i in range(N - 1):
    # Position update: t_{i+1} = t_i + R_i @ v_i
    for dim in range(d):
        # Compute the dim'th element of the matrix vector product R_i @ v_i
        rotation_times_velocity = sum(R[i][dim, j] * v[i][j] for j in range(d))
        constraint_binding = prog.AddConstraint(t[i + 1][dim] == t[i][dim] + rotation_times_velocity)
        
        add_constraint_to_qcqp(constraint_binding)
    
    # Rotation update: R_{i+1} = R_i @ Omega_i
    for row in range(d):
        for col in range(d):
            # Compute the (row, col) element of the matrix multiplication R_i @ Omega_i
            rotation_element = 0
            for j in range(d):
                rotation_element += R[i][row, j] * Omega[i][j, col]
            constraint_binding = prog.AddConstraint(R[i + 1][row, col] == rotation_element)

            add_constraint_to_qcqp(constraint_binding)

# 2. SO(3) Constraints: R_i^T @ R_i == I_d
for i in range(N):
    for row in range(d):
        for col in range(d):
            if row == col:
                # Diagonal entries
                constraint_binding_R = prog.AddConstraint(R[i].T[row, :].dot(R[i][:, col]) == 1)
                constraint_binding_Omega = prog.AddConstraint(Omega[i].T[row, :].dot(Omega[i][:, col]) == 1)
            else:
                # Off-diagonal entries
                constraint_binding_R = prog.AddConstraint(R[i].T[row, :].dot(R[i][:, col]) == 0)
                constraint_binding_Omega = prog.AddConstraint(Omega[i].T[row, :].dot(Omega[i][:, col]) == 0)
                
            add_constraint_to_qcqp(constraint_binding_R)
            add_constraint_to_qcqp(constraint_binding_Omega)


# Cost Function
# Cost is of the form: x^T Q_cost x
Q_cost = np.zeros((prog.num_vars(), prog.num_vars()))

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
        
        add_cost_to_qcqp(cost_binding)
        
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
    
    add_cost_to_qcqp(cost_binding)

# 3. Angular Velocity Differences
for i in range(N - 1):
    # Omega_{i+1} - Omega_i, flattened
    Omega_diff = [Omega[i + 1][j, l] - Omega[i][j, l] for j in range(d) for l in range(d)]
    
    # Quadratic form: Omega_diff^T * Sigma_omega * Omega_diff
    quad_form_omega = 0.0
    for r in range(d**2):
        for c in range(d**2):
            quad_form_omega += Omega_diff[r] * Sigma_omega[r, c] * Omega_diff[c]
    
    cost_binding = prog.AddCost(quad_form_omega)
    
    add_cost_to_qcqp(cost_binding)


# Set initial guesses and Solve
for i in range(N):
    prog.SetInitialGuess(t[i], t_guess[i])
    prog.SetInitialGuess(v[i], t_guess[i])
    prog.SetInitialGuess(R[i], R_guess[i])
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
        v_sol.append(result.GetSolution(v[i]))
        R_sol.append(result.GetSolution(R[i]))
        Omega_sol.append(result.GetSolution(Omega[i]))
    for k in range(K):
        p_sol.append(result.GetSolution(p[k]))
    
    visualize_results(N, K, t_sol, v_sol, R_sol, p_sol)
    
else:
    print("solve failed.")



################################################################################
##### CONVEX SDP RELAXATION
################################################################################

prog_sdp = MathematicalProgram()

# Homogenize X; i.e. X = [x, 1]^T [x, 1]
# ⌈  X   x ⌉
# ⌊ x^T  1 ⌋
X = prog_sdp.NewContinuousVariables(prog.num_vars()+1, prog.num_vars()+1, "X")
print(f"X shape: {np.shape(X)}")
X_flat = X.flatten()

# Homogenize cost matrix Q (add a row & column of zeros)
# ⌈  Q   0 ⌉
# ⌊ 0^T  0 ⌋
Q_cost = np.block([[Q_cost, np.zeros((Q_cost.shape[0], 1))], [np.zeros((1, Q_cost.shape[1] + 1))]])  # Drake is faster if we flatten first, instead of using np.trace()

# Trace(QX) Cost
Q_cost_flat = Q_cost.flatten()
prog_sdp.AddLinearCost(Q_cost_flat @ X_flat)

# Trace(QX) + b^T x + c = 0 Constraints
print(f"Number of constraints in SDP: {len(Q_constraints)}")
for i in range(len(Q_constraints)):
    # Build the b vector and c scalar into the Q matrix
    # ⌈    Q     1/2 b ⌉
    # ⌊ 1/2 b^T    c   ⌋    
    Q_constraint = np.block([
        [Q_constraints[i],                      0.5 * b_constraints[i][:, np.newaxis]],
        [0.5 * b_constraints[i][np.newaxis, :],                      c_constraints[i]]
    ])
    
    # print(Q_constraint)
    
    Q_constraint_flat = Q_constraint.flatten()
    prog_sdp.AddLinearEqualityConstraint(Q_constraint_flat @ X_flat == 0)  # Drake is faster if we flatten first, instead of using np.trace()

# X ⪰ 0 Constraint
prog_sdp.AddPositiveSemidefiniteConstraint(X)

sdp_solver_options = SolverOptions()
mosek_solver = MosekSolver()
if not mosek_solver.available():
    print("WARNING: MOSEK unavailable.")
print("Beginning SDP Solve.")
start = time.time()
result = mosek_solver.Solve(prog_sdp, solver_options=sdp_solver_options)
print(f"SDP Solve Time: {time.time() - start}")
print(f"Solved using: {result.get_solver_id().name()}")

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
    
    visualize_results(N, K, t_sol, v_sol, R_sol, p_sol)
    
else:
    print("solve failed.")
    print(f"{result.get_solution_result()}")
    print(f"{result.GetInfeasibleConstraintNames(prog_sdp)}")
    for constraint_binding in result.GetInfeasibleConstraints(prog_sdp):
        print(f"{constraint_binding.variables()}")