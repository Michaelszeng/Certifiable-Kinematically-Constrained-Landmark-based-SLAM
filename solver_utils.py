import numpy as np

def generate_ground_truth(num_timesteps, true_lin_vel, true_ang_vel):
    lin_pos = np.zeros((num_timesteps, 3))
    ang_pos = np.zeros((num_timesteps, 3, 3))
    ang_pos[0] = np.eye(3)

    for i in range(1, num_timesteps):
        lin_pos[i] = lin_pos[i-1] + ang_pos[i-1] @ true_lin_vel
        ang_pos[i] = ang_pos[i-1] @ true_ang_vel

    return lin_pos, ang_pos

def generate_measurements(true_lin_pos, true_ang_pos, true_landmarks):
    measurements = {i : dict() for i in range(len(true_landmarks))}
    for t, (lin, ang) in enumerate(zip(true_lin_pos, true_ang_pos)):
        pose = np.zeros((4, 4))
        pose[:3, :3] = ang.T
        pose[:3, 3] = -ang.T @ lin
        pose[3, 3] = 1
        for i, lm in enumerate(true_landmarks):
            measurements[i][t] = (pose @ np.hstack((lm, 1)))[:3].reshape((3, 1))
    return measurements

def print_results(ang_vel, ang_pos, landmarks, lin_vel, lin_pos, rank):
    np.set_printoptions(threshold=np.inf, suppress=True,
        formatter={'float_kind':'{:0.4f}'.format})

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

