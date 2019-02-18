from tensorflow.python.keras.models import load_model
import tensorflow as tf

from match import Match

import pyximport;
pyximport.install();
import opt.mcts_opt;

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    modelFactory = opt.mcts_opt.OPTFactory(6, load_model("../models/alphazerolike/azval.tf"),
                                           load_model("../models/alphazerolike/azpol.tf"), iterations=1, max_depth=1)
    modelFactoryTwo = modelFactory

    won, lost, drawn = 0, 0, 0
    for _ in range(5):
        for _ in range(5):
            match = Match(modelFactory, modelFactoryTwo)
            match.play()
            one, two = match.playerOne.state.score()
            if one > two:
                won += 1
            elif two > one:
                lost += 1
            else:
                drawn += 1
            match.save("DEEP_VS_DEEP")
            match.playerOne.save()

            print(".", end="")
        print()
        print(won, lost, drawn)
