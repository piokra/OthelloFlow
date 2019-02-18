from typing import Tuple

from board import Board
import abc


class Strategy:
    __metaclass__ = abc.ABCMeta

    def __init__(self, player, board: Board):
        self.player = player
        self.state = board

    def opponent_move(self, x, y):
        self.state.make_a_move(x, y, -self.player)

    @abc.abstractmethod
    def return_a_move(self) -> Tuple[int, int]:
        """Returns a move for current board state (x, y)"""
        return

    def make_a_move(self) -> Tuple[int, int]:
        move = self.return_a_move()
        x, y = move
        self.state.make_a_move(x, y, self.player)
        return move
