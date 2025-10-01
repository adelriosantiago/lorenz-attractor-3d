import numpy as np
import matplotlib.pyplot as plt
import imageio
import random

spaces = []

# ---------- Dynamical System to Test ----------
dsTest = """def formula(state, params):
    prev_dx, prev_dy, prev_dz = state
    dx = params[0] * prev_dy + params[1] * prev_dz
    dy = params[2] * prev_dx - params[3] * prev_dy
    dz = prev_dx * params[4] - params[5] * prev_dz + params[6] * prev_dy
    return np.array([dx, dy, dz])
"""

exec(dsTest)


def generate_attractor(params, steps=8000, dt=0.01):
    """Integrate the Lorenz system and return trajectory points."""
    state = np.array([1.0, 1.0, 1.0])  # initial conditions
    points = []
    for i in range(steps):
        state = state + formula(state, params) * dt
        if i > 500:  # skip transient
            points.append(state.copy())
    return np.array(points)


# ---------- Space Measurement ----------
def measure_space(points):
    """Volume of the bounding box of the trajectory."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    volume = np.prod(maxs - mins)
    return volume


# ---------- Local Search ----------
def local_search_lorenz_with_snapshots(iterations=1500):
    """Run a simple hill climb and store snapshots of each iteration."""
    params = [random.uniform(-100, 100) for _ in range(30)]
    best_points = generate_attractor(params)
    best_space = measure_space(best_points)
    best_params = params
    snapshots = [(best_params, best_points)]

    for i in range(iterations):
        # Perturb parameter array
        new_params = [p + random.uniform(-1.5, 1.5) for p in params]
        points = generate_attractor(new_params)
        space = measure_space(points)

        # Add space to line chart data
        spaces.append(space)

        if space > best_space:
            best_space = space
            best_params = params
            best_points = points
            params = new_params

        snapshots.append((params, best_points.copy()))

    return snapshots


# ---------- Main ----------
if __name__ == "__main__":
    iterations = 500
    snapshots = local_search_lorenz_with_snapshots(iterations=iterations)

    # Pick evenly spaced snapshots including the last
    num_plots = 30
    indices = np.linspace(0, iterations - 1, num_plots, dtype=int)
    selected_snapshots = [snapshots[i] for i in indices]

    # ---------- Compute global bounds ----------
    all_points = np.vstack([snap[1] for snap in selected_snapshots])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)

    xlim = (mins[0], maxs[0])
    ylim = (mins[1], maxs[1])
    zlim = (mins[2], maxs[2])

    # Make a GIF animation
    with imageio.get_writer(
        "./experiments/lorenz_local_search.gif", mode="I", duration=0.5
    ) as writer:
        for idx in range(num_plots):
            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"})
            params, points = selected_snapshots[idx]
            ax.plot(points[:, 0], points[:, 1], points[:, 2], lw=0.5, color="blue")
            ax.set_title(
                f"Snapshot {params} at Iteration {indices[idx]+1}/{iterations}",
                fontsize=10,
            )

            # Set same axis limits for all plots
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            plt.tight_layout()
            # Save to a temporary file
            temp_filename = f"./experiments/_temp_{idx}.png"
            plt.savefig(temp_filename)
            plt.close(fig)

            # Read the image and append to GIF
            image = imageio.imread(temp_filename)
            writer.append_data(image)

    # Plot the space over iterations
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(0, iterations), spaces, marker="o", markersize=2)
    ax.set_title("Space of Lorenz Attractor Over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Space (Volume of Bounding Box)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save the figure
    fig.savefig("./experiments/lorenz_space_over_iterations.png", dpi=300)
