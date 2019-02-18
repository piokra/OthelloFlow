import tensorflow as tf
from tensorflow.python.keras.models import load_model

from match_history import MatchHistory


class DeeperEvaluator:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(3, (3, 3), input_shape=(6, 6, 1), padding='SAME'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(3, (3, 3), padding='SAME'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(3, (5, 5), padding='SAME'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(3, (5, 5), padding='SAME'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(3, (3, 3), padding='SAME'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(3, (5, 5), padding='SAME'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation=tf.nn.tanh)
        ])


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    model = None
    try:
        model = load_model("deepeval.tf")
    except:
        model = DeeperEvaluator().model

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    while True:
        x, y = MatchHistory.to_training_data_ex(
            MatchHistory.sample(500, cls="MODEL_VS_MODEL", dbfile="../matchups/matches.db"))
        print("SHAPE: ", x.shape, y.shape)
        model.fit(x, y, epochs=10)
        model.save("deepeval.tf")
