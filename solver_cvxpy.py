import cvxpy as cp
import numpy as np
import pandas as pd

def solver(measurements, N, K, d, verbose=False, tol=1e-6, cov_v=1, cov_omega=1, cov_meas=1):
    # [Omega R p t]
    dim_x = 21 * N + 3 * K - 8
    dim_v = 3 * N - 3
    
    # Want to minimize tr(QX) + v^TPv
    X = cp.Variable((dim_x, dim_x), PSD=True)
    v = cp.Variable(dim_v)
    Q = np.zeros((dim_x, dim_x))
    P = np.zeros((dim_v, dim_v))

    # Add constant linear velocity objective
    P[0:dim_v, 0:dim_v] += cov_v * np.identity(dim_v)
    P[3:dim_v, 0:dim_v-3] -= cov_v * np.identity(dim_v-3)
    P[0:dim_v-3, 3:dim_v] -= cov_v * np.identity(dim_v-3)
    P[3:dim_v-3, 3:dim_v-3] += cov_v * np.identity(dim_v-6)

    # Add constant angular velocity objective
    Q[0:9*N-9, 0:9*N-9] += cov_omega * np.identity(9*N-9)
    Q[9:9*N-9, 0:9*N-18] -= cov_omega * np.identity(9*N-18)
    Q[0:9*N-18, 9:9*N-9] -= cov_omega * np.identity(9*N-18)
    Q[9:9*N-18, 9:9*N-18] += cov_omega * np.identity(9*N-27)

    for k, lm_meas in measurements.items():
        for t, meas in lm_meas.items():
            t_idx = 9*(N+t-1)
            lm_idx = 18*N-9+3*k
            trans_idx = 18*N-9+3*(K+t)

            # Add rotation objective
            for i in range(3):
                Q[t_idx+3*i:t_idx+3*(i+1), t_idx+3*i:t_idx+3*(i+1)] += cov_meas * (meas @ meas.T)

            # Add rotation to landmark objective
            for i in range(3):
                Q[lm_idx+i, t_idx+3*i:t_idx+3*(i+1)] -= cov_meas * meas.flatten()
                Q[t_idx+3*i:t_idx+3*(i+1), lm_idx+i] -= cov_meas * meas.flatten()

            # Add rotation to translation objective
            for i in range(3):
                Q[trans_idx+i, t_idx+3*i:t_idx+3*(i+1)] += cov_meas * meas.flatten()
                Q[t_idx+3*i:t_idx+3*(i+1), trans_idx+i] += cov_meas * meas.flatten()

            # Add landmark to translation objective
            Q[lm_idx:lm_idx+3, trans_idx:trans_idx+3] -= cov_meas * np.identity(3)
            Q[trans_idx:trans_idx+3, lm_idx:lm_idx+3] -= cov_meas * np.identity(3)

            # Add landmark and translation objectives
            Q[lm_idx:lm_idx+3, lm_idx:lm_idx+3] += cov_meas * np.identity(3)
            Q[trans_idx:trans_idx+3, trans_idx:trans_idx+3] += cov_meas * np.identity(3)

    constraints = []

    # Start point has 0 rotation (identity matrix)
    for i in range(9*N-9, 9*N):
        for j in range(9*N-9, dim_x):
            A = np.zeros((dim_x, dim_x))
            A[i, j] = A[j, i] = 1 if i == j else 0.5
            if i % 9 in {0, 4, 8} and j % 9 in {0, 4, 8} and 9*N-9 <= j < 9*N:
                A[-1, -1] = -1
                constraints.append(cp.trace(A @ X) == 0)
            elif i % 9 not in {0, 4, 8} or 9*N-9 <= j < 9*N:
                constraints.append(cp.trace(A @ X) == 0)

    # Start point has 0 translation
    for i in range(18*N-9+3*K, 18*N-6+3*K):
        for j in range(dim_x):
            A = np.zeros((dim_x, dim_x))
            A[i,j] = A[j,i] = 1
            constraints.append(cp.trace(A @ X) == 0)

    for t in range(N):
        for i in range(3):
            for j in range(3):
                # RR^T=I constraints
                A = np.zeros((dim_x, dim_x))
                for k in range(3):
                    A[9*(N+t-1)+3*i+k, 9*(N+t-1)+3*j+k] = 1 if 3*i+k == 3*j+k else 0.5
                    A[9*(N+t-1)+3*j+k, 9*(N+t-1)+3*i+k] = 1 if 3*i+k == 3*j+k else 0.5
                if i == j:
                    A[-1, -1] = -1
                constraints.append(cp.trace(A @ X) == 0)

                # R^TR=I constraints
                A = np.zeros((dim_x, dim_x))
                for k in range(3):
                    A[9*(N+t-1)+i+3*k, 9*(N+t-1)+j+3*k] = 1 if i+3*k == j+3*k else 0.5
                    A[9*(N+t-1)+j+3*k, 9*(N+t-1)+i+3*k] = 1 if i+3*k == j+3*k else 0.5
                if i == j:
                    A[-1, -1] = -1
                constraints.append(cp.trace(A @ X) == 0)
 
    for t in range(N):
        for i in range(3):
            for j in range(3):
                # OmegaOmega^T=I constraints
                A = np.zeros((dim_x, dim_x))
                for k in range(3):
                    A[9*t+3*i+k, 9*t+3*j+k] = 1 if 3*i+k == 3*j+k else 0.5
                    A[9*t+3*j+k, 9*t+3*i+k] = 1 if 3*i+k == 3*j+k else 0.5
                if i == j:
                    A[-1, -1] = -1
                constraints.append(cp.trace(A @ X) == 0)

                # Omega^TOmega=I constraints
                A = np.zeros((dim_x, dim_x))
                for k in range(3):
                    A[9*t+i+3*k, 9*t+j+3*k] = 1 if i+3*k == j+3*k else 0.5
                    A[9*t+j+3*k, 9*t+i+3*k] = 1 if i+3*k == j+3*k else 0.5
                if i == j:
                    A[-1, -1] = -1
                constraints.append(cp.trace(A @ X) == 0)

    # Translation odometry constraints
    for t in range(N - 1):
        for j in range(3):
            A = np.zeros((dim_x, dim_x))
            offsets = [(0, -9), (1, -6), (2, -3)]  # Offsets for R and t indices
            for offset_t, offset_R in offsets:
                A[9*(N+t)+offset_R+j, 18*N+3*(K+t)+offset_t-9] = 0.5  # R_i * t_i
                A[9*(N+t)+offset_R+j, 18*N+3*(K+t)+offset_t-6] = -0.5 # -R_i * t_{i+1}
                A[18*N+3*(K+t)+offset_t-9, 9*(N+t)+offset_R+j] = 0.5
                A[18*N+3*(K+t)+offset_t-6, 9*(N+t)+offset_R+j] = -0.5
            d = np.zeros((dim_v, 1))
            d[3*t+j,0] = 1 # v_i
            constraints.append(cp.trace(A @ X) + d.T @ v == 0)

    # Rotation odometry constraints
    for t in range(N - 2):
        for j in range(3):
            for i in range(3):
                A = np.zeros((dim_x, dim_x))
                offsets = [(-9, 0), (-8, 3), (-7, 6)]       # Offsets for R_t, Omega_t
                next_offsets = [(9, 9), (10, 10), (11, 11)] # Offsets for R_{t+2}, Omega_{t+1}
                for offset_r, offset_o in offsets:
                    A[9*(N+t)+offset_r+3*j, 9*t+offset_o+i] = 0.5
                    A[9*t+offset_o+i, 9*(N+t)+offset_r+3*j] = 0.5
                for offset_r, offset_o in next_offsets:
                    A[9*(N+t)+offset_r+3*j, 9*t+offset_o+3*i] = -0.5
                    A[9*t+offset_o+3*i, 9*(N+t)+offset_r+3*j] = -0.5
                constraints.append(cp.trace(A @ X) == 0)

    # Homogeneous constraints
    A = np.zeros((dim_x, dim_x))
    A[-1, -1] = 1
    constraints.append(cp.trace(A @ X) == 1)
    for i in range(9*N-9, 9*N):
        A = np.zeros((dim_x, dim_x))
        A[-1, i] = 0.5
        A[i, -1] = 0.5
        if i % 9 in {0, 4, 8}:
            A[-1, -1] = 1
        constraints.append(cp.trace(A @ X) == 0)

    # Redundant rotation odometry constraints
    for t in range(N - 1):
        for j in range(3):
            for i in range(3):
                A = np.zeros((dim_x, dim_x))
                offsets = [(-9, 0), (-8, 3), (-7, 6)] # Offset for R_t, Omega_t
                for offset_r, offset_o in offsets:
                    A[9*(N+t)+offset_r+3*j, 9*t+offset_o+i] = 0.5
                    A[9*t+offset_o+i, 9*(N+t)+offset_r+3*j] = 0.5
                A[-1, 9*(N+t)+3*j+i] = -0.5
                A[9*(N+t)+3*j+i, -1] = -0.5
                constraints.append(cp.trace(A @ X) == 0)
    
    # Problem definition
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X) + cp.quad_form(v, P)), constraints)
    prob.solve(solver=cp.MOSEK, verbose=verbose)
    
    # Reconstruct x
    U, S, _ = np.linalg.svd(X.value[:-1,:-1], hermitian=True)
    x = U[:, 0] * np.sqrt(S[0])
    if x[9*N-9] < 0:
        x = -x
        
    # Retrieve Omega, R, p, t
    lin_vel = v.value.reshape((N-1), 3)
    ang_vel = x[:9*N-9].reshape((N-1, 3, 3))
    ang_pos = x[9*N-9:18*N-9].reshape((N, 3, 3))
    landmarks = x[18*N-9:18*N-9+3*K].reshape((K, 3))
    lin_pos = x[18*N-9+3*K:21*N+3*K-9].reshape((N, 3))

    # Calculate rank of X
    rank = np.linalg.matrix_rank(X.value[:-1,:-1], tol=tol, hermitian=True)

    # Save results
    X.value[np.abs(X.value) < 1e-3] = 0
    DF = pd.DataFrame(X.value)
    DF.to_csv("output_files/results.csv")

    return ang_vel, ang_pos, landmarks, lin_vel, lin_pos, rank, S

