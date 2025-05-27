import numpy as np

goal_points = np.random.uniform(-3, 3, (200, 2))
robot_points = np.random.uniform(-1.5, 1.5, (200, 2))

np.save('goal_set115.npy', goal_points)
np.save('robot_set115.npy', robot_points)

print('Files generated')
