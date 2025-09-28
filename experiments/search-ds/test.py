import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import matplotlib

matplotlib.use("Agg")  # must do before importing pyplot
import matplotlib.pyplot as plt


def generate_plot(points, filename="3d_plot_output.png"):
    # Build segments for Line3DCollection
    segments = []
    for arr in arrays:
        segs = np.array([arr[:-1], arr[1:]]).transpose(1, 0, 2)  # consecutive pairs
        segments.extend(segs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    lc = Line3DCollection(segments, colors="blue", alpha=0.1, linewidths=0.5)
    ax.add_collection(lc)

    plt.savefig(filename, dpi=300)


# Suppose we have N sets, each with M points
N = 10
M = 3000
arrays = [
    np.random.rand(M, 3) + i * 0.01 for i in range(N)
]  # example data with offsets
generate_plot(arrays)
