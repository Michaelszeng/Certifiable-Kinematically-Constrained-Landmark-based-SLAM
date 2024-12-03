import numpy as np

def compute_relaxation_gap(y_bar, t, v, R, p, Omega, t_gt, v_gt, R_gt, p_gt, Omega_gt, Sigma_p, Sigma_v, Sigma_omega):
    """
    Computes the relaxation gap between the solved values and ground truth values.
    
    Parameters:
        N (int): Number of time steps.
        K (int): Number of landmarks.
        y_bar, t, v, R, p, Omega: Solved values for y_bar, t, v, R, p, and Omega.
        t_gt, v_gt, R_gt, p_gt, Omega_gt: Ground truth values for t, v, R, p, and Omega.
        Sigma_p, Sigma_v, Sigma_omega: Covariance matrices for landmark residuals, velocity, and angular velocity differences.
        
    Returns:
        float: Relaxation gap (ground truth cost - solved cost).
    """
    N = 1
    for lm_meas in y_bar.values():
        for timestep in lm_meas.keys():
            N = max(N, timestep + 1)
    K = len(y_bar)
    
    d = len(Omega[0]) 
    
    # Handle both integer and matrix covariances
    if isinstance(Sigma_p, int):
        Sigma_p = Sigma_p * np.eye(d)
    if isinstance(Sigma_v, int):
        Sigma_v = Sigma_v * np.eye(d)
    if isinstance(Sigma_omega, int):
        Sigma_omega = Sigma_omega * np.eye(d*d)
        
    print(v_gt)
    print(Omega_gt)
        
    # Account for the fact that test_generator.py only passes us one velocity since it generates tests with constant velocity
    if v_gt.ndim == 1:
        v_gt = np.array([list(v_gt)] * (N-1))
    if Omega_gt.ndim == 2:
        Omega_gt = np.array([list(Omega_gt)] * (N-1))
        
    print(v_gt)
    print(Omega_gt)
    
    def compute_cost(N, K, y_bar, t, v, R, p, Omega, Sigma_p, Sigma_v, Sigma_omega):
        cost = 0.0
        
        # 1. Landmark Residuals
        for k in range(K):
            for j, y_bar_kj in y_bar[k].items():
                # R[j] @ y_bar[k][j]
                Rj_y = (R[j] @ y_bar_kj).reshape((3,1))
                
                # (p[k] - t[j])
                p_minus_t = (p[k] - t[j]).reshape((3,1))
                
                # Residual: R[j] @ y_bar[k][j] - (p[k] - t[j])
                residual = Rj_y - p_minus_t
                
                # Quadratic form: residual^T * Sigma_p * residual
                quad_form = residual.T @ Sigma_p @ residual
                cost += quad_form

        # 2. Velocity Differences
        for i in range(N - 2):
            # v_{i+1} - v_i
            v_diff = (v[i + 1] - v[i]).reshape((3,1))
            
            # Quadratic form: v_diff^T * Sigma_v * v_diff
            quad_form_v = v_diff.T @ Sigma_v @ v_diff
            cost += quad_form_v

        # 3. Angular Velocity Differences
        for i in range(N - 2):
            # Flatten Omega difference
            Omega_diff = (Omega[i + 1] - Omega[i]).flatten().reshape((9,1))
            
            # Quadratic form: Omega_diff^T * Sigma_omega * Omega_diff
            quad_form_omega = Omega_diff.T @ Sigma_omega @ Omega_diff
            cost += quad_form_omega

        return cost

    solved_cost = compute_cost(N, K, y_bar, t, v, R, p, Omega, Sigma_p, Sigma_v, Sigma_omega)
    # print(f"solved_cost: {solved_cost}")
    ground_truth_cost = compute_cost(N, K, y_bar, t_gt, v_gt, R_gt, p_gt, Omega_gt, Sigma_p, Sigma_v, Sigma_omega)
    # print(f"ground_truth_cost: {ground_truth_cost}")

    return ground_truth_cost[0][0] - solved_cost[0][0]


def compute_mean_errors(t, v, R, p, Omega, t_gt, v_gt, R_gt, p_gt, Omega_gt):
    pass