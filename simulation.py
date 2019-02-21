import numpy as np

from os import chdir, mkdir
from image_processing import plot_particles


def simulate_sequences(number_of_sequences, sequence_length, units=1,
                       smoke=False):
    time_step = 0.02
    wind_speed_variability = 0.25
    wind_angle_variability = 0.5 * np.pi
    cloud_diffusivity = 0.005
    smoke_diffusivity = 0.01

    simulation_sequences = []

    for _ in range(number_of_sequences):
        number_of_clouds = np.random.poisson(3)
        number_of_smoke_plumes = np.random.geometric(0.75) * smoke

        clouds = []
        smoke_plumes = []

        initial_wind_speed = np.random.uniform(1, 2)
        initial_wind_angle = np.random.uniform(0, 2 * np.pi)

        for _ in range(number_of_clouds):
            cloud = {
                'location': np.random.uniform(-units, units, 2),
                'spread': 0.1,
                'new_particle_limit_g': 1000,
                'new_particle_limit_p': 75
            }

            clouds.append(cloud)

        for _ in range(number_of_smoke_plumes):
            smoke_plume = {
                'base_location': np.random.uniform(-units, units, 2),
                'source_weights': [0.5, 1],
                'source_separation': 2,
                'source_delay': 3,
                'new_particle_limit': 100
            }

            smoke_plumes.append(smoke_plume)

        simulation = RadarSimulation2D(sequence_length, time_step, clouds,
                                       smoke_plumes, cloud_diffusivity,
                                       smoke_diffusivity, initial_wind_speed,
                                       initial_wind_angle,
                                       wind_speed_variability,
                                       wind_angle_variability)

        simulation_sequence = simulation.run()
        simulation_sequences.append(simulation_sequence)

    return simulation_sequences


class RadarSimulation2D:

    dimensionality = 2

    def __init__(self, sequence_length, time_step, clouds,
                 smoke_plumes, cloud_diffusivity, smoke_diffusivity,
                 initial_wind_speed, initial_wind_angle,
                 wind_speed_variability, wind_angle_variability):
        self.sequence_length = sequence_length
        self.time_step = time_step
        self.clouds = clouds
        self.smoke_plumes = smoke_plumes
        self.cloud_diffusivity = cloud_diffusivity
        self.smoke_diffusivity = smoke_diffusivity
        self.initial_wind_speed = initial_wind_speed
        self.initial_wind_angle = initial_wind_angle
        self.wind_speed_variability = wind_speed_variability
        self.wind_angle_variability = wind_angle_variability

        self.wind_sequence = []
        self.cloud_sources = []
        self.smoke_sources = []

    def run(self):
        self.populate_wind_sequence()

        self.populate_cloud_sources()
        self.populate_smoke_sources()

        cloud_particles = ParticleContainer(self.dimensionality,
                                            self.cloud_diffusivity)
        smoke_particles = ParticleContainer(self.dimensionality,
                                            self.smoke_diffusivity)

        simulation_sequence = []

        for wind_vector in self.wind_sequence:
            for cloud_source in self.cloud_sources:
                new_cloud_particles = cloud_source.emit()
                cloud_particles.add_particles(new_cloud_particles)

            for smoke_source in self.smoke_sources:
                new_smoke_particles = smoke_source.emit()
                smoke_particles.add_particles(new_smoke_particles)

            cloud_particles.update_positions(wind_vector, self.time_step)
            smoke_particles.update_positions(wind_vector, self.time_step)

            simulation = np.concatenate((smoke_particles.particles,
                                         cloud_particles.particles))
            simulation_sequence.append(simulation)

        return simulation_sequence

    def populate_wind_sequence(self):
        speed = self.initial_wind_speed
        angle = self.initial_wind_angle
        speed_variance = self.time_step * self.wind_speed_variability
        angle_variance = self.time_step * self.wind_angle_variability

        for frame in range(self.sequence_length):
            speed += np.random.normal(0, speed_variance)
            angle += np.random.normal(0, angle_variance)
            wind_velocity = speed * np.array([np.cos(angle), np.sin(angle)])

            self.wind_sequence.append(wind_velocity)

    def populate_cloud_sources(self):
        for cloud in self.clouds:
            location = cloud['location']
            spread = cloud['spread']
            stop_time = 2

            coin_toss = np.random.uniform() > 0.5

            if coin_toss:
                high = cloud['new_particle_limit_g']
                cloud_source = GaussianSource(location, spread, high=high,
                                              stop_time=stop_time)
            else:
                high = cloud['new_particle_limit_p']
                cloud_source = PointSource(location, high=high,
                                           stop_time=stop_time)

            self.cloud_sources.append(cloud_source)

    def populate_smoke_sources(self):
        for smoke_plume in self.smoke_plumes:
            base_location = smoke_plume['base_location']
            source_weights = smoke_plume['source_weights']
            source_separation = smoke_plume['source_separation']
            source_delay = smoke_plume['source_delay']
            high = smoke_plume['new_particle_limit']

            for j, w in enumerate(source_weights):
                wind_sum = sum(self.wind_sequence[:j * source_separation])
                separation = self.time_step * wind_sum
                location = base_location + separation
                smoke_source = PointSource(location, high=w * high,
                                           start_time=j * source_delay)

                self.smoke_sources.append(smoke_source)


class Source:

    def __init__(self, location, low=0, high=100, start_time=0,
                 stop_time=np.inf):
        self.location = location
        self.low = low
        self.high = high
        self.start_time = start_time
        self.stop_time = stop_time

        self.time = 0

    def amount(self):
        self.time += 1

        if self.start_time < self.time < self.stop_time:
            amount = np.random.randint(self.low, self.high)
        else:
            amount = 0

        return amount


class PointSource(Source):

    def emit(self):
        amount = self.amount()

        particles = np.tile(self.location, (amount, 1))

        return particles


class GaussianSource(Source):

    def __init__(self, location, spread, low=0, high=100, start_time=0,
                 stop_time=np.inf):
        super().__init__(location, low, high, start_time, stop_time)
        self.spread = spread

    def emit(self):
        amount = self.amount()

        mean = self.location
        d = mean.size
        random_matrix = np.random.rand(d, d)
        cov = self.spread * random_matrix@random_matrix.T

        particles = np.random.multivariate_normal(mean, cov, amount)

        return particles


class ParticleContainer:

    def __init__(self, dimensionality, diffusivity):
        self.diffusivity = diffusivity

        self.particles = np.zeros((0, dimensionality))

    def add_particles(self, new_particles):
        self.particles = np.concatenate((self.particles, new_particles))

    def update_positions(self, velocity, time_step):
        n, d = self.particles.shape
        mean = time_step * velocity
        cov = self.diffusivity * time_step * np.identity(d)

        self.particles += np.random.multivariate_normal(mean, cov, n)


def test():
    n = 10

    sequence_length = 16
    units = 1

    print('Simulating sequences...')

    cloud = simulate_sequences(n, sequence_length, units, smoke=False)
    smoke = simulate_sequences(n, sequence_length, units, smoke=True)

    test_sequences = [('Cloud', cloud), ('Smoke', smoke)]

    print('Saving sequences...')

    for category in test_sequences:
        category_name, sequences = category

        mkdir(category_name)
        chdir(category_name)

        for i, sim in enumerate(sequences):
            path = '%05d' % (i + 1)

            mkdir(path)
            chdir(path)

            for j, smoke_plume in enumerate(sim):
                name = path + '_' + '%02d' % (j + 1)
                plot_particles(smoke_plume, units, name=name)

            chdir('..')

        chdir('..')


if __name__ == '__main__':
    test()
