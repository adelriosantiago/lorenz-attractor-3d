import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import matplotlib

# matplotlib.use("Agg")  # Must occur before importing pyplot
import matplotlib.pyplot as plt


def generate_plot(points, filename="3d_plot_output.png"):
    # Build segments for Line3DCollection
    segments = []

    # Shift each dataset farther apart
    for i, arr in enumerate(points):
        # shift each set by 2*MAX_ABS*i in X (or any axis you want)
        # shift = np.array([2 * MAX_ABS, 0, 0])  # shifting along X axis
        # shifted_arr = arr + shift
        shifted_arr = arr

        # Build segments for Line3DCollection
        segs = np.array([shifted_arr[:-1], shifted_arr[1:]]).transpose(1, 0, 2)
        segments.extend(segs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    lc = Line3DCollection(segments, colors="blue", alpha=0.3, linewidths=0.5)
    ax.add_collection(lc)

    # Set axis limits from lims
    max_x, max_y, max_z = np.array(points).reshape(-1, 3).max(axis=0)
    ax.set_xlim(-max_x, max_x)
    ax.set_ylim(-max_y, max_y)
    ax.set_zlim(-max_z, max_z)

    # Label axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
    # plt.savefig(filename, dpi=300)
