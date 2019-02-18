import datetime
import os
import time
from math import sqrt
from multiprocessing import Pool
from tensorflow.python.keras import losses
from tensorflow.python.keras.models import load_model, clone_model, Model, save_model
import tensorflow as tf
from tqdm import tqdm
from match import Match
from matchups.util import sample_mcts_data
import pyximport;

pyximport.install();
import opt.mcts_opt;

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _training(valm: Model, polm: Model):
    boards, vals, pols = sample_mcts_data(1024, 256, "../matchups/matches.db")
    valm.compile(loss='mse', optimizer='adam', metrics=['mse'])
    polm.compile(loss=losses.categorical_crossentropy, optimizer='adam', metrics=[losses.categorical_crossentropy])
    valm.fit(boards, vals, epochs=8)
    polm.fit(boards, pols, epochs=8)


def _compete(old_models, new_models, size=64):
    models = [*old_models, *new_models]
    for i, model in enumerate(models):
        save_model(model, str(i) + '.tmp')

    with Pool(1) as pool:
        wins = sum(pool.starmap(_compete_partial, [([0, 1], [2, 3], 16)] * 4))

    for i in range(4):
        os.remove(str(i) + '.tmp')

    return 0.55 * size <= wins


def _compete_partial(old_models, new_models, l=64):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.logging.set_verbosity(tf.logging.FATAL)

    old_models = map(lambda x: load_model(str(x)+'.tmp'), old_models)
    new_models = map(lambda x: load_model(str(x)+'.tmp'), new_models)
    print("Competing")
    modelFactory = opt.mcts_opt.OPTFactory(6, *new_models, iterations=40, max_depth=4)
    modelFactoryTwo = opt.mcts_opt.OPTFactory(6, *old_models, iterations=40, max_depth=4)

    won, lost, drawn = 0, 0, 0
    for _ in tqdm(range(l)):
        match = Match(modelFactory, modelFactoryTwo)
        match.play()
        one, two = match.playerOne.state.score()
        if one > two:
            won += 1
        elif two > one:
            lost += 1
        else:
            drawn += 1

    return won


def _provide_new_training_data(models, size=64):
    for i, model in enumerate(models):
        save_model(model, str(i) + '.tmp')
    with Pool(1) as pool:
        pool.starmap(_provide_new_training_data_partial, [([0, 1], 32)] * 4)

    for i in range(2):
        os.remove(str(i) + '.tmp')


def _provide_new_training_data_partial(models, l=128):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.logging.set_verbosity(tf.logging.FATAL)

    models = map(lambda x: load_model(str(x)+'.tmp'), models)
    print("Training data")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    modelFactory = opt.mcts_opt.OPTFactory(6, *models, iterations=40, max_depth=4)
    modelFactoryTwo = modelFactory

    won, lost, drawn = 0, 0, 0
    for _ in tqdm(range(l)):
        match = Match(modelFactory, modelFactoryTwo)
        match.play()
        match.playerOne.save("../matchups/matches.db")


def learn(iterations=100):
    current_best_val = load_model("../models/alphazerolike/azval.tf")
    current_best_pol = load_model("../models/alphazerolike/azpol.tf")
    candidate_val = clone_model(current_best_val)
    candidate_pol = clone_model(current_best_pol)
    _provide_new_training_data((candidate_val, candidate_pol))
    for _ in tqdm(range(iterations)):
        candidate_val = clone_model(current_best_val)
        candidate_pol = clone_model(current_best_pol)

        _training(candidate_val, candidate_pol)

        if _compete((current_best_val, current_best_pol), (candidate_val, candidate_pol)):
            save_model(candidate_val, "../models/alphazerolike/azval.tf")
            save_model(candidate_pol, "../models/alphazerolike/azpol.tf")
            _provide_new_training_data((candidate_val, candidate_pol))

            save_model(candidate_val,
                       "../models/alphazerolike/history/" + datetime.datetime.now().strftime("%Y_%m_%d%H_%M"))
            save_model(candidate_pol,
                       "../models/alphazerolike/history/" + datetime.datetime.now().strftime("%Y_%m_%d%H_%M"))

            current_best_pol = candidate_pol
            current_best_val = candidate_val


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.logging.set_verbosity(tf.logging.FATAL)
    learn()
