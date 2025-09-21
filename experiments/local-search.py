import numpy as np

# ---------- Lorenz system ----------
def lorenz(state, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def generate_attractor(sigma, rho, beta, steps=20000, dt=0.01):
    state = np.array([1.0, 1.0, 1.0])  # initial conditions
    points = []
    for i in range(steps):
        state = state + lorenz(state, sigma, rho, beta) * dt
        if i > 1000:  # skip transient
            points.append(state.copy())
    return np.array(points)

# ---------- Space measurement ----------
def measure_space(points):
    # bounding box volume
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    volume = np.prod(maxs - mins)
    return volume

# ---------- Local search ----------
def local_search_lorenz(iterations=50):
    # Start with standard parameters
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    best_points = generate_attractor(sigma, rho, beta)
    best_space = measure_space(best_points)
    best_params = (sigma, rho, beta)

    for i in range(iterations):
        # Perturb parameters
        new_sigma = sigma + np.random.uniform(-1, 1)
        new_rho   = rho   + np.random.uniform(-2, 2)
        new_beta  = beta  + np.random.uniform(-0.1, 0.1)

        # Keep parameters positive to stay stable
        new_sigma = max(0.1, new_sigma)
        new_rho = max(0.1, new_rho)
        new_beta = max(0.01, new_beta)

        # Generate new attractor and measure space
        points = generate_attractor(new_sigma, new_rho, new_beta)
        space = measure_space(points)

        # Hill climbing acceptance
        if space > best_space:
            best_space = space
            best_params = (new_sigma, new_rho, new_beta)
            best_points = points
            sigma, rho, beta = new_sigma, new_rho, new_beta
            print(f"Iteration {i+1}: New best space {space:.2f} with params {best_params}")

    return best_params, best_points, best_space

# ---------- Run it ----------
best_params, best_points, best_space = local_search_lorenz(iterations=30)

print("\nBest parameters found:", best_params)
print("Bounding box volume of attractor:", best_space)
