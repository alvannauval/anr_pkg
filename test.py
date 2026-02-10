import numpy as np
import matplotlib.pyplot as plt

def rx(theta):
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(theta), -np.sin(theta), 0],
                     [0, np.sin(theta), np.cos(theta), 0],
                     [0, 0, 0, 1]])

def ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                     [0, 1, 0, 0],
                     [-np.sin(theta), 0, np.cos(theta), 0],
                     [0, 0, 0, 1]])

def rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                     [np.sin(theta), np.cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def translation(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T

# 1. Define Points
origin = np.array([0, 0, 0, 1])

# 2. Origin -> P1: Rotate 45 deg around Z, then move 5 units along X
T_0_1 = rz(np.radians(45)) @ translation(5, 0, 0)
p1 = T_0_1 @ origin

# 3. P1 -> P2: Rotate 60 deg around Y, then move 4 units along X
T_1_2 = ry(np.radians(60)) @ translation(4, 0, 0)
p2 = T_0_1 @ T_1_2 @ origin # Chained transformation

# --- Visualization ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract coordinates
# pts = np.array([origin, p1])
pts = np.array([origin, p1, p2])
x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

# Plot the arm
ax.plot(x, y, z, 'o-', color='navy', lw=4, markersize=8)

# Formatting the 3D space
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_xlim([-7, 7]); ax.set_ylim([-7, 7]); ax.set_zlim([-7, 7])

plt.show()