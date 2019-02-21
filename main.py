from simulation import simulate_sequences
from data_processing import construct_still_data_set, construct_clip_data_set
from models import still_model, clip_model

t = 1000  # Number of training sequences per class.
v = 100  # Number of validation sequences per class.

sequence_length = 16
units = 1
bins_per_unit = 32

print('Simulating sequences...')

cloud = simulate_sequences(t + v, sequence_length, units, smoke=False)
smoke = simulate_sequences(t + v, sequence_length, units, smoke=True)

print('Constructing data sets...')

x_still_t, y_still_t = construct_still_data_set(cloud[:t], smoke[:t],
                                                bins_per_unit=bins_per_unit,
                                                units=units)
x_still_v, y_still_v = construct_still_data_set(cloud[-v:], smoke[-v:],
                                                bins_per_unit=bins_per_unit,
                                                units=units)
x_clip_t, y_clip_t = construct_clip_data_set(cloud[:t], smoke[:t],
                                             bins_per_unit=bins_per_unit,
                                             units=units)
x_clip_v, y_clip_v = construct_clip_data_set(cloud[-v:], smoke[-v:],
                                             bins_per_unit=bins_per_unit,
                                             units=units)

epochs = 5
batch_size = 50
number_of_classes = 2

still = still_model(number_of_classes, x_still_v.shape[1:])
clip = clip_model(number_of_classes, x_clip_v.shape[1:])

still.fit(x_still_t, y_still_t, batch_size=batch_size, epochs=epochs,
          validation_data=(x_still_v, y_still_v))
clip.fit(x_clip_t, y_clip_t, batch_size=batch_size, epochs=epochs,
         validation_data=(x_clip_v, y_clip_v))

still_scores = still.evaluate(x_still_v, y_still_v)
clip_scores = clip.evaluate(x_clip_v, y_clip_v)

print('STILL ACCURACY:', still_scores[1])
print('CLIP ACCURACY:', clip_scores[1])
