from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# Flatten initial guesses into a parameter vector
x0 = np.hstack([
    np.array(t_guess).flatten(),
    np.array(R_guess).flatten(),
    np.array(v_guess).flatten(),
    np.array(Omega_guess).flatten(),
    np.array(p_guess).flatten(),
])

# Define the residual function
def residuals(x):
    # Extract variables from x
    t = x[:N*d].reshape(N, d)
    R_flat = x[N*d:2*N*d].reshape(N, d, d)
    v = x[2*N*d:3*N*d].reshape(N, d)
    Omega = x[3*N*d:4*N*d].reshape(N, d)
    p = x[4*N*d:].reshape(K, d)
    
    res = []
    
    # Observation residuals
    for k in range(K):
        for j, y in y_bar[k].items():
            res.append(R_flat[j] @ y - (t[j] - p[k]))
    
    # Dynamic constraints
    for i in range(N - 1):
        res.append(t[i + 1] - t[i] - v[i])
        res.append(R_flat[i + 1] - R_flat[i] @ np.eye(d) + Omega[i])  # Simplified
    
    # Return as a flat array
    return np.concatenate([r.flatten() for r in res])

# Run Levenberg-Marquardt
result = least_squares(residuals, x0, method='lm')

# Extract optimized variables
x_opt = result.x
t_opt = x_opt[:N*d].reshape(N, d)
R_opt = x_opt[N*d:2*N*d].reshape(N, d, d)
v_opt = x_opt[2*N*d:3*N*d].reshape(N, d)
Omega_opt = x_opt[3*N*d:4*N*d].reshape(N, d)
p_opt = x_opt[4*N*d:].reshape(K, d)
