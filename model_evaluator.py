from typing import Tuple

from board import Board
from evaluator import Evaluator

import tensorflow as tf

from match import Match


class ModelEvaluator(Evaluator):
    model: tf.keras.Model

    def __init__(self, player, model):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        self.flip = player != 1
        if type(model) is str:
            self.model = tf.keras.models.load_model(model)
        else:
            self.model = model

    def evaluate(self, start: Board, move: Tuple[int, int], end: Board):
        board = end.board
        if self.flip:
            board = -board

        dx, dy = board.shape
        board.shape = (1, dx, dy, 1)
        val, = self.model.predict(board)
        r = (val / 2 + 0.5) ** 2
        return r[0]


if __name__ == "__main__":
    from random_strategy import RandomStrategy
    from strategy_factory import StrategyFactory

    model = ModelEvaluator(1, "models/6x6eval.tf")
    match = Match(StrategyFactory(RandomStrategy), StrategyFactory(RandomStrategy))

    match.play()
    score = match.playerOne.state.score()
    print(score)
    end = match.playerOne.state
    print(model.evaluate(None, None, end))