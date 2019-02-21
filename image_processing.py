import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

from data_processing import particles_to_array

bom_colours = np.array(
    [
        [0, 0, 0],
        [245, 245, 255],
        [179, 182, 253],
        [118, 126, 251],
        [0, 50, 251],
        [50, 215, 195],
        [32, 149, 143],
        [18, 102, 101],
        [255, 252, 56],
        [253, 198, 46],
        [252, 149, 38],
        [251, 101, 32],
        [250, 20, 27],
        [196, 13, 18],
        [117, 4, 7],
        [39, 0, 1],
    ]
)

m = 1.5
e = 2
n = len(bom_colours)
boundaries = [m * i**e for i in range(n + 1)]

colour_map = matplotlib.colors.ListedColormap(bom_colours/255)
boundary_norm = matplotlib.colors.BoundaryNorm(boundaries, n, clip=True)


def plot_particles(particles, size=1, bpu=32, ppb=8, name=None):
    h = particles_to_array(particles, bpu, size)

    fig = plt.figure(figsize=(1, 1), dpi=2*ppb*bpu*size, frameon=False)
    axes = plt.Axes(fig, [0, 0, 1, 1])
    axes.set_axis_off()
    fig.add_axes(axes)

    axes.imshow(h, origin='lower', cmap=colour_map, norm=boundary_norm)

    if name:
        fig.savefig(name)
    else:
        fig.show()

    plt.close(fig)
