import tensorflow as tf

from match_history import MatchHistory


class Simple6x6Evaluator:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(3, (3, 3), input_shape=(6, 6, 1), padding='SAME'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(3, (3, 3), padding='SAME'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(3, (3, 3), padding='SAME'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(1, (3, 3), padding='SAME'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    model = Simple6x6Evaluator().model
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    x, y = MatchHistory.to_training_data(MatchHistory.sample(2500, cls="MCTS_VS_MCTS", dbfile="../matchups/matches.db"))
    model.fit(x, y, epochs=10)
    model.save("6x6eval.tf")
