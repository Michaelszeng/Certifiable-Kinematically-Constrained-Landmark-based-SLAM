import numpy as np

def generate_ground_truth(num_timesteps, true_lin_vel, true_ang_vel):
    lin_pos = np.zeros((num_timesteps, 3))
    ang_pos = np.zeros((num_timesteps, 3, 3))
    ang_pos[0] = np.eye(3)

    for i in range(1, num_timesteps):
        lin_pos[i] = lin_pos[i-1] + ang_pos[i-1] @ true_lin_vel
        ang_pos[i] = ang_pos[i-1] @ true_ang_vel

    return lin_pos, ang_pos

def generate_measurements(true_lin_pos, true_ang_pos, true_landmarks, noise=0, dropout=0):
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

