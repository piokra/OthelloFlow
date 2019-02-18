import numpy as np
from typing import Tuple

from board import Board
from strategy import Strategy
from random import choice


class RandomStrategy(Strategy):

    def __init__(self, player, board: Board):
        super().__init__(player, board)

    def return_a_move(self) -> Tuple[int, int]:
        x, y = np.where(self.state.board == 0)
        valid_xy = filter(lambda _xy: self.state.is_move_valid(self.player, *_xy), zip(x, y))
        return choice(list(valid_xy))


if __name__ == "__main__":
    gameOne = Board()
    gameTwo = Board()
    randomPlayerOne = RandomStrategy(1, gameOne)
    randomPlayerTwo = RandomStrategy(-1, gameTwo)
    while gameOne.has_valid_move(1) or gameTwo.has_valid_move(-1):
        if gameOne.has_valid_move(1):
            move = randomPlayerOne.make_a_move()
            randomPlayerTwo.opponent_move(*move)

        if gameTwo.has_valid_move(-1):
            move = randomPlayerTwo.make_a_move()
            randomPlayerOne.opponent_move(*move)
        print(gameOne.score())
