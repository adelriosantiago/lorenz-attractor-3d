import numpy as np
import matplotlib.pyplot as plt
import imageio



# Define the dynamical system to test
dsTest = """def formula(state, params):
    dx = params[0] * (state[1] - state[0])
    dy = state[0] * (params[1] - state[2]) - state[1]
    dz = state[0] * state[1] - params[2] * state[2]
    return np.array([dx, dy, dz])
"""
exec(dsTest)

def tryGenerate(params, steps=3000, dt=0.01):
    state = np.array([1.0, 1.0, 1.0])  # initial conditions
    points = []
    for i in range(steps):
        state = state + formula(state, params) * dt
        if i > steps // 10:  # skip transient
            points.append(state.copy())
    return np.array(points)

# Check attractor generation
points = tryGenerate([10.0, 28.0, 8.0/3.0])

# Visualize the attractor
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(points[:,0], points[:,1], points[:,2])
plt.show()


