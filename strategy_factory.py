from board import Board
from strategy import Strategy


class StrategyFactory:
    def __init__(self, strategyClass, **kwargs):
        self.strategyClass = strategyClass
        self.kwargs = kwargs

    def build(self, player: int, board: Board) -> Strategy:
        return self.strategyClass(player, board, **self.kwargs)
