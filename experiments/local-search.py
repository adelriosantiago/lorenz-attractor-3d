import numpy as np
import matplotlib.pyplot as plt

# ---------- Lorenz system ----------
def lorenz(state, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def generate_attractor(sigma, rho, beta, steps=8000, dt=0.01):
    """Integrate the Lorenz system and return trajectory points."""
    state = np.array([1.0, 1.0, 1.0])  # initial conditions
    points = []
    for i in range(steps):
        state = state + lorenz(state, sigma, rho, beta) * dt
        if i > 500:  # skip transient
            points.append(state.copy())
    return np.array(points)

# ---------- Space measurement ----------
def measure_space(points):
    """Volume of the bounding box of the trajectory."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    volume = np.prod(maxs - mins)
    return volume

# ---------- Local search with saving results ----------
def local_search_lorenz_with_snapshots(iterations=10):
    """Run a simple hill climb and store snapshots of each iteration."""
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    best_points = generate_attractor(sigma, rho, beta)
    best_space = measure_space(best_points)
    best_params = (sigma, rho, beta)
    snapshots = [(best_params, best_points)]

    for i in range(iterations-1):
        # Perturb parameters
        new_sigma = sigma + np.random.uniform(-1, 1)
        new_rho   = rho   + np.random.uniform(-2, 2)
        new_beta  = beta  + np.random.uniform(-0.1, 0.1)
        new_sigma = max(0.1, new_sigma)
        new_rho = max(0.1, new_rho)
        new_beta = max(0.01, new_beta)

        points = generate_attractor(new_sigma, new_rho, new_beta)
        space = measure_space(points)

        if space > best_space:
            best_space = space
            best_params = (new_sigma, new_rho, new_beta)
            best_points = points
            sigma, rho, beta = new_sigma, new_rho, new_beta

        snapshots.append(((sigma, rho, beta), best_points.copy()))

    return snapshots

# ---------- Main ----------
if __name__ == "__main__":
    snapshots = local_search_lorenz_with_snapshots(iterations=10)

    # Plotting snapshots: 10 rows × 1 column
    fig, axes = plt.subplots(10, 1, figsize=(10, 60), subplot_kw={'projection':'3d'})
    plt.subplots_adjust(hspace=0.5)

    for idx, ax in enumerate(axes):
        params, points = snapshots[idx]
        ax.plot(points[:,0], points[:,1], points[:,2], lw=0.5, color='blue')
        ax.set_title(f"Iteration {idx+1}: σ={params[0]:.2f}, ρ={params[1]:.2f}, β={params[2]:.2f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.tight_layout()
    plt.show()
