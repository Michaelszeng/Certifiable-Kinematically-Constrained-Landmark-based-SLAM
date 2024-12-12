import numpy as np

def generate_ground_truth(num_timesteps, true_lin_vel, true_ang_vel):
    lin_pos = np.zeros((num_timesteps, 3))
    ang_pos = np.zeros((num_timesteps, 3, 3))
    ang_pos[0] = np.eye(3)

    for i in range(1, num_timesteps):
        lin_pos[i] = lin_pos[i-1] + ang_pos[i-1] @ true_lin_vel
        if true_ang_vel.shape == (3, 3):
            ang_pos[i] = ang_pos[i-1] @ true_ang_vel
        else:
            ang_pos[i] = ang_pos[i-1] @ true_ang_vel[i-1]

    return lin_pos, ang_pos

def generate_measurements(true_lin_pos, true_ang_pos, true_landmarks, noise=0.0, dropout=0.0):
    measurements = {i : dict() for i in range(len(true_landmarks))}
    for t, (lin, ang) in enumerate(zip(true_lin_pos, true_ang_pos)):
        pose = np.zeros((4, 4))
        pose[:3, :3] = ang.T
        pose[:3, 3] = -ang.T @ lin
        pose[3, 3] = 1
        for i, lm in enumerate(true_landmarks):
            if np.random.rand() >= dropout:
                true_meas = (pose @ np.hstack((lm, 1)))[:3]
                measurements[i][t] = (true_meas + np.random.normal(0, noise, 3)).reshape((3, 1))
    return measurements

def print_ground_truth(ang_vel, ang_pos, landmarks, lin_vel, lin_pos):
    np.set_printoptions(threshold=np.inf, suppress=True,
        formatter={'float_kind':'{:0.4f}'.format})

    print("Ground truth angular velocity:")
    print(ang_vel)
    print()

    print("Ground truth angular position:")
    print(ang_pos)
    print()

    print("Ground truth landmarks:")
    print(landmarks)
    print()

    print("Ground truth linear velocity:")
    print(lin_vel)
    print()

    print("Ground truth linear position:")
    print(lin_pos)
    print()

def print_results(ang_vel, ang_pos, landmarks, lin_vel, lin_pos, rank, S):
    np.set_printoptions(threshold=np.inf, suppress=True,
        formatter={'float_kind':'{:0.4f}'.format})

    print("Singular values:")
    print(S)
    print()

    print("Angular velocity:")
    print(ang_vel)
    print()

    print("Angular position:")
    print(ang_pos)
    print()

    print("Landmarks:")
    print(landmarks)
    print()

    print("Linear velocity:")
    print(lin_vel)
    print()

    print("Linear position:")
    print(lin_pos)
    print()

    print("Rank of X is", rank)
    print()

def generate_test_file(file_path, measurements, true_lin_pos, true_lin_vel, true_landmarks, true_ang_pos, true_ang_vel):
    with open(file_path, 'w') as f:
        f.write("import numpy as np\n\n")
        f.write("d = 3   # dimension of space (3D)\n\n")

        # Write measurements
        f.write("# Measurement data: maps landmark to {timestamp: measurement} dicts\n")
        f.write("y_bar = {\n")
        for lm, lm_measurements in measurements.items():
            f.write(f"    {lm}: {{\n")
            for t, measurement in lm_measurements.items():
                measurement_str = np.array2string(measurement.flatten(), separator=',')
                f.write(f"        {t}: np.array([{measurement_str}]).T,\n")
            f.write("    },\n")
        f.write("}\n\n")

        # Write N and K
        f.write("N = 1\n")
        f.write("for lm_meas in y_bar.values():\n")
        f.write("    for timestep in lm_meas.keys():\n")
        f.write("        N = max(N, timestep + 1)\n")
        f.write("K = len(y_bar)\n\n")

        # Write covariance matrices
        f.write("# Covariances\n")
        f.write("Sigma_p = np.linalg.inv(4*np.eye(d))  # Covariance matrix for position\n")
        f.write("Sigma_v = np.linalg.inv(np.eye(d))  # Covariance matrix for velocity\n")
        f.write("Sigma_omega = np.linalg.inv(np.eye(d**2))  # Covariance matrix for angular velocity\n\n")

        # Write initial guesses
        f.write("# Initial guesses:\n")
        f.write("t_guess = [\n")
        for pos in true_lin_pos:
            f.write(f"    {pos.tolist()},\n")
        f.write("]\n")

        f.write("R_guess = [\n")
        for ang in true_ang_pos:
            f.write(f"    np.array({ang.tolist()}),\n")
        f.write("]\n")

        # Write linear velocity guess
        f.write("v_guess = [\n")
        for _ in range(len(true_lin_pos) - 1):  # N - 1
            f.write(f"    {true_lin_vel.tolist()},\n")
        f.write("]\n")

        # Write angular velocity guess
        f.write("Omega_guess = [\n")
        for _ in range(len(true_ang_pos) - 1):  # N - 1
            f.write(f"    np.array({true_ang_vel.tolist()}),\n")
        f.write("]\n")

        f.write("p_guess = [\n")
        for lm in true_landmarks:
            f.write(f"    {lm.tolist()},\n")
        f.write("]\n")

def generate_measurements_moving_landmarks(true_lin_pos, true_ang_pos, true_landmarks, true_landmark_velocities, noise=0.0, dropout=0.0):
    measurements = {i : dict() for i in range(len(true_landmarks))}
    for t, (lin, ang) in enumerate(zip(true_lin_pos, true_ang_pos)):
        pose = np.zeros((4, 4))
        pose[:3, :3] = ang.T
        pose[:3, 3] = -ang.T @ lin
        pose[3, 3] = 1
        for i, lm in enumerate(true_landmarks):
            lm_current = lm + t*true_landmark_velocities[i]
            if np.random.rand() >= dropout:
                true_meas = (pose @ np.hstack((lm_current, 1)))[:3]
                measurements[i][t] = (true_meas + np.random.normal(0, noise, 3)).reshape((3, 1))
    return measurements

def print_ground_truth_moving_landmarks(ang_vel, ang_pos, landmarks, lin_vel, lin_pos, landmark_velocities):
    np.set_printoptions(threshold=np.inf, suppress=True,
        formatter={'float_kind':'{:0.4f}'.format})

    print("Ground truth linear position:")
    print(lin_pos)
    print()
    
    print("Ground truth linear velocity:")
    print(lin_vel)
    print()
    
    print("Ground truth landmarks:")
    print(landmarks)
    print()

    print("Ground truth angular position:")
    print(ang_pos)
    print()
    
    print("Ground truth angular velocity:")
    print(ang_vel)
    print()
    
    print("Ground truth landmark velocities:")
    print(landmark_velocities)
    print()
    
    
def generate_test_file_moving_landmarks(file_path, measurements, true_lin_pos, true_lin_vel, true_landmarks, true_ang_pos, true_ang_vel, true_landmark_vel):
    with open(file_path, 'w') as f:
        f.write("import numpy as np\n\n")
        f.write("d = 3   # dimension of space (3D)\n\n")

        # Write measurements
        f.write("# Measurement data: maps landmark to {timestamp: measurement} dicts\n")
        f.write("y_bar = {\n")
        for lm, lm_measurements in measurements.items():
            f.write(f"    {lm}: {{\n")
            for t, measurement in lm_measurements.items():
                measurement_str = np.array2string(measurement.flatten(), separator=',')
                f.write(f"        {t}: np.array([{measurement_str}]).T,\n")
            f.write("    },\n")
        f.write("}\n\n")

        # Write N and K
        f.write("N = 1\n")
        f.write("for lm_meas in y_bar.values():\n")
        f.write("    for timestep in lm_meas.keys():\n")
        f.write("        N = max(N, timestep + 1)\n")
        f.write("K = len(y_bar)\n\n")

        # Write covariance matrices
        f.write("# Covariances\n")
        f.write("Sigma_p = np.linalg.inv(4*np.eye(d))  # Covariance matrix for position\n")
        f.write("Sigma_v = np.linalg.inv(np.eye(d))  # Covariance matrix for velocity\n")
        f.write("Sigma_omega = np.linalg.inv(np.eye(d**2))  # Covariance matrix for angular velocity\n\n")

        # Write initial guesses
        f.write("# Initial guesses:\n")
        f.write("t_guess = [\n")
        for pos in true_lin_pos:
            f.write(f"    {pos.tolist()},\n")
        f.write("]\n")

        f.write("R_guess = [\n")
        for ang in true_ang_pos:
            f.write(f"    np.array({ang.tolist()}),\n")
        f.write("]\n")

        # Write linear velocity guess
        f.write("v_guess = [\n")
        for _ in range(len(true_lin_pos) - 1):  # N - 1
            f.write(f"    {true_lin_vel.tolist()},\n")
        f.write("]\n")

        # Write angular velocity guess
        f.write("Omega_guess = [\n")
        for _ in range(len(true_ang_pos) - 1):  # N - 1
            f.write(f"    np.array({true_ang_vel.tolist()}),\n")
        f.write("]\n")

        f.write("p_guess = [\n")
        for lm in true_landmarks:
            f.write(f"    {lm.tolist()},\n")
        f.write("]\n")
        
        f.write("z_guess = [\n")
        for z in true_landmark_vel:
            f.write(f"    {z.tolist()},\n")
        f.write("]\n")
