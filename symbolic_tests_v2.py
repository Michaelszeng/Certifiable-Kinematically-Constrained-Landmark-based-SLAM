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

current_folder = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(current_folder, "test_data")
sys.path.append(test_data_path)
from test8 import *

np.set_printoptions(edgeitems=30, linewidth=270, precision=4, suppress=True)

# Full State:
#      d*N d*(N-1) d*K, d*d*N, d*d*(N-1)
# x = [ t,    v,    p,    R,    Omega ]


################################################################################
##### NON-CONVEX PROGRAM
################################################################################

prog = MathematicalProgram()

# Variable Definitions
# NOTE: DEFINE THESE IN THE ORDER THEY APPEAR IN OUR FULL STATE REPRESENTATION
t = [prog.NewContinuousVariables(d, f"t_{i}") for i in range(N)]                # Positions t_i
v = [prog.NewContinuousVariables(d, f"v_{i}") for i in range(N-1)]                # Velocities v_i
p = [prog.NewContinuousVariables(d, f"p_{k}") for k in range(K)]                # Landmark positions p_k
R = [prog.NewContinuousVariables(d, d, f"R_{i}") for i in range(N)]             # Rotations R_i
Omega = [prog.NewContinuousVariables(d, d, f"Ω_{i}") for i in range(N-1)]     # Angular velocities Ω_i


def add_constraint_to_qcqp(name, constraint_binding):
    """
    Helper function to format a generic (quadratic) constraint into QCQP form by
    adding to the `Q_constraints`, `b_constraints`, and `c_constraits` lists.
    
    Args:
        constraint_binding: Binding<Constraint> object containing binding for the added constraint.
        
    Returns:
        None; augments `Q_constraints`, `b_constraints`, and `c_constraints` directly.
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
        
    constraint_names.append(name)
    Q_constraints.append(Q_constraint)
    b_constraints.append(b_constraint)
    c_constraints.append(c_constraint)
        
    

def add_cost_to_qcqp(cost_binding):
    """
    Helper function to format a generic (quadratic) cost into QCQP form by
    adding to the `Q_cost` matrix.
    
    Note that it is assumed the optimization admits a least squares formulation,
    so there are no linear terms in the cost.
    
    TODO: build compatibility with linear costs as well to make this work with
    more general formulations that might have linear constraints.
    
    Args:
        cost_binding: Binding<Cost> object containing binding for the added cost.
        
    Returns:
        None; augments `Q_cost` directly.
    """
    cost = cost_binding.evaluator()
    cost_vars = cost_binding.variables()
        
    for j, v1 in enumerate(cost_vars):
        v1_idx = prog.FindDecisionVariableIndex(v1)

        for l, v2 in enumerate(cost_vars):
            v2_idx = prog.FindDecisionVariableIndex(v2)

            Q_cost[v1_idx, v2_idx] += cost.Q()[j, l]
    

# Constraint Definitions
# Each constraint is of the form: 1/2 x^T Q_constraints[i] x + b_constraints[i]^T x + c_constraints[i] = 0
constraint_names = []  # For debugging convenience
Q_constraints = []
b_constraints = []
c_constraints = []

# 1. Linear Odometry Constraint
for i in range(N - 1):
    for dim in range(d):
        # LINEAR T, QUADRATIC R,V FORMULATION: t_{i+1} = t_i + R_i @ v_i
        # Compute the dim'th element of the matrix vector product R_i @ v_i
        rotation_times_velocity = sum(R[i][dim, j] * v[i][j] for j in range(d))
        constraint_binding = prog.AddConstraint(t[i + 1][dim] == t[i][dim] + rotation_times_velocity)
        
        add_constraint_to_qcqp(f"t_odom_{i}_{dim}", constraint_binding)
        
        
        # LINEAR V, QUADRATIC R,T FORMULATION: R_i^T @ t_{i+1} = R_i^T @ t_i + v_i
        # Compute the dim'th element of the matrix vector product R_i @ t_{i+1}
        R_t_i_plus_1 = rotation_times_velocity = sum(R[i].T[dim, j] * t[i+1][j] for j in range(d))
        # Compute the dim'th element of the matrix vector product R_i @ t_i
        R_t_i = rotation_times_velocity = sum(R[i].T[dim, j] * t[i][j] for j in range(d))
        
        constraint_binding = prog.AddConstraint(R_t_i_plus_1 == R_t_i + v[i][dim])
        
        add_constraint_to_qcqp(f"t_odom_rotated_{i}_{dim}", constraint_binding)
    
# 2. Rotational Odometry Constraint
for i in range(N - 1):
    for row in range(d):
        for col in range(d):
            # LINEAR R_{i+1}, QUADRATIC R_i, Omega_i: R_{i+1} = R_i @ Omega_i
            # Compute the (row, col) element of the matrix multiplication R_i @ Omega_i
            rotation_element = 0
            for j in range(d):
                rotation_element += R[i][row, j] * Omega[i][j, col]
            constraint_binding = prog.AddConstraint(R[i + 1][row, col] == rotation_element)

            add_constraint_to_qcqp(f"R_odom_{i}_{row}_{col}", constraint_binding)
            
            # LINEAR R_i, QUDRATIC R_{i+1}, Omega_i: R_{i+1} @ Omega_i^T = R_i
            # Compute the (row, col) element of the matrix multiplication R_{i+1} @ Omega_i^T
            rotation_element = 0
            for j in range(d):
                rotation_element += R[i+1][row, j] * Omega[i].T[j, col]
            constraint_binding = prog.AddConstraint(rotation_element == R[i][row, col])

            add_constraint_to_qcqp(f"R_odom_rotated_{i}_{row}_{col}", constraint_binding)
            
    if i < N - 2:
        for row in range(d):
            for col in range(d):
                # ALL QUADRATIC TERMS: R_i @ Omega_i = R_{i+2} @ Omega_{i+1}^T
                # Compute the (row, col) element of the matrix multiplication R_i @ Omega_i
                left_side = 0
                right_side = 0
                for j in range(d):
                    left_side += R[i][row, j] * Omega[i][j, col]
                    right_side += R[i+2][row, j] * Omega[i].T[j, col]
                constraint_binding = prog.AddConstraint(left_side == right_side)

                add_constraint_to_qcqp(f"R_odom_substituted_{i}_{row}_{col}", constraint_binding)

# 3. SO(3) Constraint on Rotation: R_i^T @ R_i == I_d and R_i @ R_i^T == I_d
for i in range(N):
    for row in range(d):
        for col in range(d):
            if row == col:
                # Diagonal entries
                constraint_binding1 = prog.AddConstraint(R[i].T[row, :].dot(R[i][:, col]) == 1)
                constraint_binding2 = prog.AddConstraint(R[i][row, :].dot(R[i].T[:, col]) == 1)
            else:
                # Off-diagonal entries
                constraint_binding1 = prog.AddConstraint(R[i].T[row, :].dot(R[i][:, col]) == 0)
                constraint_binding2 = prog.AddConstraint(R[i][row, :].dot(R[i].T[:, col]) == 0)
                
            add_constraint_to_qcqp(f"R_ortho1_{i}_{row}_{col}", constraint_binding1)
            add_constraint_to_qcqp(f"R_ortho2_{i}_{row}_{col}", constraint_binding2)

# 4. SO(3) Constraint on Angular Velocity: Omega_i^T @ Omega_i == I_d and Omega_i @ Omega_i^T == I_d
for i in range(N-1):
    for row in range(d):
        for col in range(d):
            if row == col:
                # Diagonal entries
                constraint_binding1 = prog.AddConstraint(Omega[i].T[row, :].dot(Omega[i][:, col]) == 1)
                constraint_binding2 = prog.AddConstraint(Omega[i][row, :].dot(Omega[i].T[:, col]) == 1)
            else:
                # Off-diagonal entries
                constraint_binding1 = prog.AddConstraint(Omega[i].T[row, :].dot(Omega[i][:, col]) == 0)
                constraint_binding2 = prog.AddConstraint(Omega[i][row, :].dot(Omega[i].T[:, col]) == 0)

            add_constraint_to_qcqp(f"omega_ortho1_{i}_{row}_{col}", constraint_binding1)
            add_constraint_to_qcqp(f"omega_ortho2_{i}_{row}_{col}", constraint_binding2)
            
# 5. Initial identity rotation:
for row in range(d):
    for col in range(d):
        # Identity matrix has the following property:
        # Product of any 2 diagonal elements == 1
        constraint_binding = prog.AddConstraint(R[0][row,row]*R[0][col,col] == 1)
        add_constraint_to_qcqp(f"R_initial_diag_{row}_{col}", constraint_binding)
        
        if row != col:
            # Off-diagonal entries
            constraint_binding = prog.AddConstraint(R[0][row,col]*R[0][col,row] == 0)
            add_constraint_to_qcqp(f"R_initial_off_diag_{row}_{col}", constraint_binding)

# 6: Initial 0 translation:
for dim in range(d):
    constraint_binding = prog.AddConstraint(t[0][dim] * t[0][dim] == 0)
    
    add_constraint_to_qcqp(f"t_initial_{row}_{col}", constraint_binding)


# Cost Function
# Cost is of the form: 1/2 x^T Q_cost x
Q_cost = np.zeros((prog.num_vars(), prog.num_vars()))

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
        
        cost_binding = prog.AddCost(quad_form)
        
        add_cost_to_qcqp(cost_binding)
        
# 2. Velocity Differences
for i in range(N - 2):
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
for i in range(N - 2):
    # Omega_{i+1} - Omega_i, flattened
    Omega_diff = [Omega[i + 1][j, l] - Omega[i][j, l] for j in range(d) for l in range(d)]
    
    # Quadratic form: Omega_diff^T * Sigma_omega * Omega_diff
    quad_form_omega = 0.0
    for r in range(d*d):
        for c in range(d*d):
            quad_form_omega += Omega_diff[r] * Sigma_omega[r, c] * Omega_diff[c]
    
    cost_binding = prog.AddCost(quad_form_omega)
    
    add_cost_to_qcqp(cost_binding)


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
    
    visualize_results(N, K, t_sol, v_sol, R_sol, p_sol, Omega_sol)
    
else:
    print("solve failed.")



################################################################################
##### CONVEX SDP RELAXATION
################################################################################

# Clean up matrices
Q_cost[np.abs(Q_cost) < 1e-9] = 0
for i in range(len(Q_constraints)):
    Q_constraints[i][np.abs(Q_constraints[i]) < 1e-9] = 0
    b_constraints[i][np.abs(b_constraints[i]) < 1e-9] = 0
    c_constraints[i][np.abs(c_constraints[i]) < 1e-9] = 0

prog_sdp = MathematicalProgram()

# Homogenize X; i.e. X = [x, 1]^T [x, 1]
# ⌈  X   x ⌉
# ⌊ x^T  1 ⌋
X = prog_sdp.NewSymmetricContinuousVariables(prog.num_vars() + 1, "X")
print(f"X shape: {np.shape(X)}")
X_flat = X.flatten()

# Homogenize cost matrix Q (add a row & column of zeros)
# ⌈  Q   0 ⌉
# ⌊ 0^T  0 ⌋
Q_cost = np.block([[0.5 * Q_cost, np.zeros((Q_cost.shape[0], 1))], [np.zeros((1, Q_cost.shape[1] + 1))]])

# Trace(QX) Cost
Q_cost_flat = Q_cost.flatten()
prog_sdp.AddLinearCost(Q_cost_flat @ X_flat)

# Trace(QX) + b^T x + c = 0 Constraints
for i in range(len(Q_constraints)):
    # Build the b vector and c scalar into the Q matrix
    # ⌈    Q     1/2 b ⌉
    # ⌊ 1/2 b^T    c   ⌋    
    Q_constraint = np.block([
        [0.5 * Q_constraints[i],                0.5 * b_constraints[i][:, np.newaxis]],
        [0.5 * b_constraints[i][np.newaxis, :],                      c_constraints[i]]
    ])
    
    Q_constraint_flat = Q_constraint.flatten()
    prog_sdp.AddLinearEqualityConstraint(Q_constraint_flat @ X_flat == 0)  # Drake is faster if we flatten first, instead of using np.trace()
    
# Tighten initial identity rotation constraint by setting everything in the rows and columns corresponding to non-diagonal components of R_0 to 0
R_0_idx = prog.FindDecisionVariableIndex(R[0][0,0])
for dim in range(R_0_idx, R_0_idx + d*d):  # Iterate through components of R_0
    for j in range(np.shape(X)[0]):
        Q = np.zeros((np.shape(X)))
        if dim-R_0_idx not in {0, 4, 8} and j-R_0_idx not in {0, 4, 8}:  # Any component of X in the R[0] rows/columns that doesn't contain a diagonal component of R[0] gets set to 0
            Q[dim,j] = Q[j,dim] = 0.5
            prog_sdp.AddConstraint(Q.flatten() @ X.flatten() == 0)

# Tighten initial 0 translation constraint by setting everything in t_0 rows and columns to 0
for dim in range(d):
    for j in range(np.shape(X)[0]):
        Q = np.zeros((np.shape(X)))
        Q[dim, j] = Q[j, dim] = 0.5
        prog_sdp.AddConstraint(Q.flatten() @ X.flatten() == 0)


# Add Homogenous constraints
# X[-1,-1] = 1
Q = np.zeros((np.shape(X)))
Q[-1,-1] = 1
prog_sdp.AddConstraint(Q.flatten() @ X.flatten() == 1)

# Anchor the the initial rotation and translation for the homogenous odometry constraints
for dim in range(R_0_idx, R_0_idx + d*d):  # Enforce first rotation as identity rotation
    Q = np.zeros((np.shape(X)))
    Q[-1, dim] = 0.5
    Q[dim, -1] = 0.5
    if dim-R_0_idx in {0, 4, 8}:
        prog_sdp.AddConstraint(Q.flatten() @ X.flatten() == 1)
    else:
        prog_sdp.AddConstraint(Q.flatten() @ X.flatten() == 0)

for i in range(d):  # Enforce first translation as 0
    Q = np.zeros((np.shape(X)))
    Q[-1, i] = 0.5
    Q[i, -1] = 0.5
    prog_sdp.AddConstraint(Q.flatten() @ X.flatten() == 0)

# X ⪰ 0 Constraint
prog_sdp.AddPositiveSemidefiniteConstraint(X)

print(f"Number of constraints in SDP: {len(prog_sdp.GetAllConstraints())}")

sdp_solver_options = SolverOptions()
mosek_solver = MosekSolver()
if not mosek_solver.available():
    print("WARNING: MOSEK unavailable.")
    
# Set Initial Guess for SDP
x_guess = np.array(sum(t_guess, []) +
                    sum(v_guess, []) +
                    sum(p_guess, []) +
                    [val for R in R_guess for val in R.T.flatten()] +
                    [val for Omega in Omega_guess for val in Omega.T.flatten()] + 
                    [1]).reshape((np.shape(X)[0], 1))
X_guess = x_guess @ x_guess.T
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        prog_sdp.SetInitialGuess(X[i, j], X_guess[i, j])

print("Beginning SDP Solve.")
start = time.time()
result = mosek_solver.Solve(prog_sdp, solver_options=sdp_solver_options)
print(f"SDP Solve Time: {time.time() - start}")
print(f"Solved using: {result.get_solver_id().name()}")

if result.is_success():
    X_sol = result.GetSolution(X)
    print(f"Rank of X: {np.linalg.matrix_rank(X_sol, rtol=1e-6, hermitian=True)}")
    
    # Save X as csv
    X_sol[np.abs(X_sol) < 1e-3] = 0
    labels = [var.get_name() for var in prog.decision_variables()][:X.shape[0]-1] + ["_"]
    DF = pd.DataFrame(X_sol, index=labels, columns=labels)
    DF.to_csv("output_files/drake_solver.csv")
    
    # Reconstruct x
    U, S, _ = np.linalg.svd(X_sol[:-1,:-1], hermitian=True)  # ignore homogenous parts of X
    x_sol = U[:, 0] * np.sqrt(S[0])
    if x_sol[R_0_idx] < 0:
        x_sol = -x_sol
    print(f"Singular Values: {S}")
        
    t_sol = []
    v_sol = []
    R_sol = []
    Omega_sol = []
    p_sol = []
    for i in range(N):
        t_sol.append(x_sol[d*i : d*(i+1)])
        R_sol.append(x_sol[d*N + d*(N-1) + d*K + d*d*i : d*N + d*(N-1) + d*K + d*d*(i+1)].reshape((3,3)).T)  # No idea why this transpose is needed
    for i in range(N-1):
        v_sol.append(x_sol[d*N + d*i : d*N + d*(i+1)])
        Omega_sol.append(x_sol[d*N + d*(N-1) + d*K + d*d*N + d*d*i : d*N + d*(N-1) + d*K + d*d*N + d*d*(i+1)].reshape((3,3)).T)  # No idea why this transpose is needed
    for k in range(K):
        p_sol.append(x_sol[d*N + d*(N-1) + d*k : d*N + d*(N-1) + d*(k+1)])
    
    visualize_results(N, K, t_sol, v_sol, R_sol, p_sol, Omega_sol)
    
else:
    print("solve failed.")
    print(f"{result.get_solution_result()}")
    print(f"{result.GetInfeasibleConstraintNames(prog_sdp)}")
    for constraint_binding in result.GetInfeasibleConstraints(prog_sdp):
        print(f"{constraint_binding.variables()}")






# x = prog_sdp.NewContinuousVariables(prog.num_vars(), "x")  # x is a vector

# # Add objective
# prog_sdp.AddQuadraticCost(x.T @ Q_cost @ x)

# # Add constraints
# for i in range(len(Q_constraints)):
#     Q_i = Q_constraints[i]
#     b_i = b_constraints[i]
#     c_i = c_constraints[i]

#     # Constraint: 1/2 x^T Q_i x + b_i^T x + c_i == 0
#     prog_sdp.AddQuadraticConstraint(Q=Q_i, b=b_i, lb=-c_i, ub=-c_i, vars=x)
    
# # Set initial guesses and Solve
# x_guess = (sum(t_guess, []) +
#             sum(v_guess, []) +
#             sum(p_guess, []) +
#             [val for R in R_guess for val in R.T.flatten()] +
#             [val for Omega in Omega_guess for val in Omega.T.flatten()])
# prog_sdp.SetInitialGuess(x, x_guess)

# result = Solve(prog_sdp)

# if result.is_success():
#     t_sol = []
#     v_sol = []
#     R_sol = []
#     Omega_sol = []
#     p_sol = []
    
#     x_sol = result.GetSolution(x)
    
#     idx = 0
    
#     # Helper function to unflatten a 3x3 matrix from column-major order
#     def unflatten_column_major(flat_matrix):
#         return np.array(flat_matrix).reshape(3, 3).T

#     t_sol = [x_sol[idx + i: idx + i + 3] for i in range(0, N * 3, 3)]
#     idx += N * 3

#     v_sol = [x_sol[idx + i: idx + i + 3] for i in range(0, (N-1) * 3, 3)]
#     idx += (N-1) * 3

#     p_sol = [x_sol[idx + i: idx + i + 3] for i in range(0, K * 3, 3)]
#     idx += K * 3

#     R_sol = [unflatten_column_major(x_sol[idx + i: idx + i + 9]) for i in range(0, N * 9, 9)]
#     idx += N * 9

#     Omega_sol = [unflatten_column_major(x_sol[idx + i: idx + i + 9]) for i in range(0, (N-1) * 9, 9)]
    
#     visualize_results(N, K, t_sol, v_sol, R_sol, p_sol, Omega_sol)
    
# else:
#     print("solve failed.")
