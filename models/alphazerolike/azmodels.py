import numpy as np

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.models import load_model

from match_history import MatchHistory
from models.alphazerolike.layers import conv_layer, residual_layer, policy_layer, value_layer
import tensorflow as tf

def policy_model(input_dim=(6, 6, 1), depth=5) -> Model:
    inp = Input(input_dim)
    conv = conv_layer(inp)
    res = conv
    for _ in range(depth):
        res = residual_layer(res)
    pol = policy_layer(res)
    return Model(inputs=inp, outputs=pol)


def value_model(input_dim=(6, 6, 1), depth=5) -> Model:
    inp = Input(input_dim)
    conv = conv_layer(inp)
    res = conv
    for _ in range(depth):
        res = residual_layer(res)
    val = value_layer(res)
    return Model(inputs=inp, outputs=val)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    vmodel = value_model()
    pmodel = policy_model()
    print(pmodel.output_shape, vmodel.output_shape)
    vmodel.compile(loss='mse', optimizer='adam', metrics=['mse'])
    pmodel.compile(loss='mse', optimizer='adam', metrics=['mse'])

    # x, y = MatchHistory.to_training_data_ex(
    # MatchHistory.sample(500, cls="MODEL_VS_MODEL", dbfile="../../matchups/matches.db"))
    # pmodel.predict(np.zeros((1,6,6,1), dtype='float64'))
    #vmodel.fit(x, y, epochs=30)
    vmodel.save("azval.tf")
    pmodel.save("azpol.tf")
