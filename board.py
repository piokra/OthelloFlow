from typing import Tuple
from mathutil import matrix_cut
from itertools import product

import numpy as np


class Board:
    def __init__(self, size=6):
        if size % 2:
            raise AssertionError("Board has to have even size")
        board = np.zeros((size, size), dtype='int8')

        board[size // 2, size // 2] = 1
        board[size // 2, size // 2 - 1] = -1
        board[size // 2 - 1, size // 2] = -1
        board[size // 2 - 1, size // 2 - 1] = 1

        self.board = board
        self.size = size

    def flip(self):
        self.board = -self.board

    def has_valid_move(self, player):
        if np.count_nonzero(self.board) == self.size * self.size:
            return False

        empty = np.where(self.board == 0)
        return any(self.is_move_valid(player, x, y) for x, y in zip(empty[0], empty[1]))

    def is_move_valid(self, player, x, y):
        if self.board[x, y] != 0:
            return False
        self.board[x, y] = player
        for dir in product((-1, 0, 1), (-1, 0, 1)):
            if self.__is_good_dir(player, (x, y), dir):
                self.board[x, y] = 0
                return True
        self.board[x, y] = 0
        return False

    def make_a_move(self, x, y, player):
        if self.is_move_valid(player, x, y):
            self.board[x, y] = player
            for dir in product((-1, 0, 1), (-1, 0, 1)):
                if self.__is_good_dir(player, (x, y), dir):
                    self.__flip_dir(player, (x, y), dir)

            return

        print(self.valid_moves(player))
        raise IndexError("Not a valid move: player: {} x: {} y: {}".format(player, x, y))

    def valid_moves(self, player):
        x, y = np.where(self.board == 0)
        valid_xy = filter(lambda _xy: self.is_move_valid(player, *_xy), zip(x, y))
        return list(valid_xy)

    def score(self):
        return np.count_nonzero(self.board == 1), np.count_nonzero(self.board == -1)

    def print(self):
        print(self.score())
        print(self.board)

    def __is_good_dir(self, player, pos, dir):
        x, y = pos
        dx, dy = dir
        met = False
        if dx == 0 and dy == 0:
            return False
        ps = 2
        t = 0
        while ps is not 0 and 0 <= x + dx * t < self.size and 0 <= y + dy * t < self.size:
            if self.board[x + dx * t, y + dy * t] == -player:
                met = True
            if self.board[x + dx * t, y + dy * t] == player:
                ps -= 1
            if self.board[x + dx * t, y + dy * t] == 0:
                return False
            t += 1
        if ps == 0:
            return met
        return False

    def __flip_dir(self, player, pos, dir):
        x, y = pos
        dx, dy = dir
        ps = 2
        t = 0

        while ps is not 0 and 0 <= x + dx * t < self.size and 0 <= y + dy * t < self.size:
            if self.board[x + dx * t, y + dy * t] == player:
                ps -= 1
            self.board[x + dx * t, y + dy * t] = player
            t += 1

    def clone(self):
        _clone = Board(size=self.size)
        _clone.board = np.copy(self.board)
        return _clone

    def __hash__(self):
        return hash(self.board.tostring())

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return (self.board == other.board).all()


if __name__ == "__main__":
    game = Board()
    player = 1
    print("Start")
    while game.has_valid_move(1) or game.has_valid_move(-1):

        game.print()
        try:
            print("Player nb:", player)
            x, y = input().split()
            x, y = int(x), int(y)
            game.make_a_move(x, y, player)
            player *= -1
        except IndexError as e:
            print(e)
