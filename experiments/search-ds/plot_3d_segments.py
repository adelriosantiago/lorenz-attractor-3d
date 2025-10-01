import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import matplotlib

# matplotlib.use("Agg")  # must do before importing pyplot
import matplotlib.pyplot as plt


def generate_plot(points, MAX_ABS=1, filename="3d_plot_output.png"):
    # Build segments for Line3DCollection
    segments = []

    # Shift each dataset farther apart
    for i, arr in enumerate(points):
        # shift each set by 2*MAX_ABS*i in X (or any axis you want)
        shift = np.array([2 * MAX_ABS, 0, 0])  # shifting along X axis
        shifted_arr = arr + shift

        # Build segments for Line3DCollection
        segs = np.array([shifted_arr[:-1], shifted_arr[1:]]).transpose(1, 0, 2)
        segments.extend(segs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    lc = Line3DCollection(segments, colors="blue", alpha=0.8, linewidths=0.5)
    ax.add_collection(lc)

    # Set axis limits from -MAX_ABS to MAX_ABS
    ax.set_xlim(-2 * MAX_ABS, 2 * MAX_ABS)
    ax.set_ylim(-MAX_ABS, MAX_ABS)
    ax.set_zlim(-MAX_ABS, MAX_ABS)

    # Label axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
    # plt.savefig(filename, dpi=300)
