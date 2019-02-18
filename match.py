from typing import Tuple, List, Any
import sqlite3
import pickle

from board import Board
from strategy import Strategy
from strategy_factory import StrategyFactory


class Match:
    history: List[Tuple[int, Tuple[int, int]]]

    def __init__(self, playerOne: StrategyFactory, playerTwo: StrategyFactory, **boardArgs):
        self.playerOne = playerOne.build(1, Board(**boardArgs))
        self.playerTwo = playerTwo.build(-1, Board(**boardArgs))
        self.history = []

    def play(self) -> Tuple[int, int]:
        while self.playerOne.state.has_valid_move(1) or self.playerTwo.state.has_valid_move(-1):

            if self.playerOne.state.has_valid_move(1):
                move = self.playerOne.make_a_move()
                self.history.append((1, move))
                self.playerTwo.opponent_move(*move)
            if self.playerTwo.state.has_valid_move(-1):
                move = self.playerTwo.make_a_move()
                self.history.append((-1, move))
                self.playerOne.opponent_move(*move)
        return self.playerOne.state.score()

    def save(self, name="ANY", first="UNKNOWN", second="UNKNOWN"):
        connection = sqlite3.connect("matches.db")
        cursor = connection.cursor()
        cursor.execute(
            "create table if not exists Matches(id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " cls VARCHAR(20), firstPlayer VARCHAR(20), secondPlayer VARCHAR(20), "
            "pickle BINARY)")

        pickled = pickle.dumps(self.history)
        cursor = connection.cursor()
        cursor.execute("insert into Matches(cls, pickle, firstPlayer, secondPlayer) values(?, ?, ?, ?)",
                       (name, pickled, first, second))
        cursor.close()
        connection.commit()
        connection.close()


if __name__ == "__main__":
    from random_strategy import RandomStrategy

    match = Match(StrategyFactory(RandomStrategy), StrategyFactory(RandomStrategy))
    print(match.play())
    match.playerOne.state.print()
    match.save("6x6")
