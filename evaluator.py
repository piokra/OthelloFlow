import abc
from abc import ABC
from typing import Tuple

from board import Board


class Evaluator:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def evaluate(self, start: Board, move: Tuple[int, int], end: Board):
        pass


class ZeroEvaluator(Evaluator):
    def evaluate(self, start: Board, move: Tuple[int, int], end: Board):
        return 0
