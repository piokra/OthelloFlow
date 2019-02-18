from tensorflow.python.keras.layers import Reshape, Flatten, Convolution2D, Dense, merge, BatchNormalization, ReLU, \
    Activation


def residual_layer(x):
    conv = Convolution2D(256, (3, 3), padding='SAME')(x)
    norm = BatchNormalization()(conv)
    relu = Activation('relu')(norm)
    conv2 = Convolution2D(256, (3, 3), padding='SAME')(relu)
    m = merge.concatenate([conv2, x])
    return Activation('relu')(m)


def conv_layer(x):
    conv = Convolution2D(256, (3, 3), padding='SAME')(x)
    norm = BatchNormalization()(conv)
    return Activation('relu')(norm)


def value_layer(x):
    conv = Convolution2D(1, (1, 1), padding='SAME')(x)
    norm = BatchNormalization()(conv)
    relu = Activation('relu')(norm)
    flat = Flatten()(relu)
    dense = Dense(256, activation='relu')(flat)
    out = Dense(1, activation='tanh')(dense)
    return out


def policy_layer(x, n=6):
    conv = Convolution2D(2, (1, 1))(x)
    norm = BatchNormalization()(conv)
    relu = Activation('relu')(norm)
    flat = Flatten()(relu)
    dense = Dense(n * n, activation='sigmoid')(flat)
    shaped = Reshape((n, n))(dense)
    return shaped
