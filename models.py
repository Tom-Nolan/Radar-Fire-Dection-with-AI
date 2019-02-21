from keras.optimizers import rmsprop
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed


def still_model(number_of_classes, input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))

    opt = rmsprop(lr=0.001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    return model


def clip_model(number_of_classes, input_shape):
    model = Sequential()

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',
                                     activation='relu'),
                              input_shape=input_shape))
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',
                                     activation='relu')))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    model.add(Dense(number_of_classes, activation='softmax'))

    opt = rmsprop(lr=0.001, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    return model
