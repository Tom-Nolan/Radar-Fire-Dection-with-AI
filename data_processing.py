import numpy as np

from keras.utils import to_categorical


def particles_to_array(particles, bins_per_unit, units):
    x, y = particles.T
    bins = 2 * bins_per_unit * units
    box = [[-units, units], [-units, units]]

    h = np.histogram2d(x, y, bins=bins, range=box)[0]

    return h


def construct_clip_data_set(*args, bins_per_unit=32, units=1):
    x = []
    y = []

    for label, category in enumerate(args):
        for sequence in category:
            array_sequence = []

            for particles in sequence:
                h = particles_to_array(particles, bins_per_unit, units)
                array_sequence.append(np.reshape(h, h.shape + (1, )))

            x.append(array_sequence)
            y.append(to_categorical(label, len(args)))

    x = np.array(x)
    y = np.array(y)

    return x, y


def construct_still_data_set(*args, bins_per_unit=32, units=1):
    x = []
    y = []

    for label, category in enumerate(args):
        for sequence in category:
            particles = sequence[-1]
            h = particles_to_array(particles, bins_per_unit, units)

            x.append(np.reshape(h, h.shape + (1,)))
            y.append(to_categorical(label, len(args)))

    x = np.array(x)
    y = np.array(y)

    return x, y
