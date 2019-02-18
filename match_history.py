import numpy as np
import pickle
import sqlite3
from typing import List, Any, Tuple, Iterable

from board import Board
from match import Match


def _qw(cursor, query, args):
    cursor.execute(query, args)
    ret = cursor.fetchall()
    cursor.close()
    return ret


class MatchHistory:
    end: Board
    boards: List[Board]
    name: str

    def __init__(self, history: List[Tuple[int, Tuple[int, int]]], name="ANY"):
        self.name = name
        start = Board()
        self.boards = list((start.clone(),))
        for player, (x, y) in history:
            start.make_a_move(x, y, player)
            self.boards.append(start.clone())

        self.end = self.boards[-1].clone()

    @staticmethod
    def load(id, dbfile="./matchups/matches.db") -> 'MatchHistory':
        db = sqlite3.connect(dbfile)
        cursor = db.cursor()
        cursor.execute("select pickle from Matches where id = ?", (id,))
        pickle, = cursor.fetchone()
        cursor.close()
        db.close()
        return MatchHistory.unpickle(pickle)

    @staticmethod
    def unpickle(pickle_string) -> 'MatchHistory':
        history = pickle.loads(pickle_string)
        return MatchHistory(history)

    @staticmethod
    def sample(n: int, cls: str = None, firstPlayer: str = None, secondPlayer: str = None,
               dbfile="./matchups/matches.db") -> Iterable['MatchHistory']:
        db = sqlite3.connect(dbfile)
        c = db.cursor()
        pickles = None
        if cls is not None:
            if firstPlayer is None:
                pickles = MatchHistory.__sample_class(c, n, cls)
            elif secondPlayer is None:
                pickles = MatchHistory.__sample_class_and_player(c, n, cls, firstPlayer)
            else:
                pickles = MatchHistory.__sample_class_and_players(c, n, cls, firstPlayer, secondPlayer)
        elif firstPlayer is not None:
            if secondPlayer is not None:
                pickles = MatchHistory.__sample_class_and_players(c, n, cls, firstPlayer, secondPlayer)
            else:
                pickles = MatchHistory.__sample_class_and_player(c, n, cls, firstPlayer)
        else:
            pickles = MatchHistory.__simple_sample(db.cursor(), n)
        db.close()
        return map(lambda x: MatchHistory.unpickle(*x), pickles)

    @staticmethod
    def to_training_data(histories: Iterable['MatchHistory']) -> Tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        for history in histories:
            b, w = history.boards[-1].score()
            y.extend([np.sign(b - w)] * len(history.boards))
            for board in history.boards:
                dx, dy = board.board.shape
                x.append(board.board.reshape((dx, dy, 1)))
        return np.array(x, dtype='int8'), np.array(y, dtype='int8')

    @staticmethod
    def to_training_data_ex(histories: Iterable['MatchHistory']) -> Tuple[np.ndarray, np.ndarray]:
        x, y = MatchHistory.to_training_data(histories)
        mx, my = -x, -y
        tx, ty = x.swapaxes(1, 2), 1*y
        mtx, mty = -tx, -ty
        return np.concatenate((x, mx, tx, mtx)), np.concatenate((y, my, ty, mty))

    @staticmethod
    def __sample_class(cursor, n: int, cls: str):
        return _qw(cursor,
                   "SELECT pickle FROM main.Matches"
                   " WHERE id IN (SELECT id FROM main.Matches WHERE cls = ? ORDER BY RANDOM() LIMIT ?)",
                   (cls, n,))

    @staticmethod
    def __sample_player(cursor: sqlite3.Cursor, n: int, player: str):
        return _qw(cursor,
                   "SELECT pickle FROM main.Matches"
                   " WHERE id IN (SELECT id FROM main.Matches "
                   "WHERE firstPlayer = ? or secondPlayer = ? "
                   "ORDER BY RANDOM() LIMIT ?)",
                   (player, player, n,))

    @staticmethod
    def __sample_players(cursor: sqlite3.Cursor, n: int, playerOne: str, playerTwo: str):
        return _qw(cursor,
                   "SELECT pickle FROM main.Matches"
                   " WHERE id IN (SELECT id FROM main.Matches "
                   "WHERE firstPlayer = ? and secondPlayer = ? "
                   "ORDER BY RANDOM() LIMIT ?)",
                   (playerOne, playerTwo, n,))

    @staticmethod
    def __sample_class_and_player(cursor: sqlite3.Cursor, n: int, cls: str, player: str):
        return _qw(cursor,
                   "SELECT pickle FROM main.Matches"
                   " WHERE id IN (SELECT id FROM main.Matches "
                   "WHERE cls = ? AND (firstPlayer = ? or secondPlayer = ?)"
                   " ORDER BY RANDOM() LIMIT ?)",
                   (cls, player, player, n,))

    @staticmethod
    def __sample_class_and_players(cursor: sqlite3.Cursor, n: int, cls: str, playerOne: str, playerTwo: str):
        return _qw(cursor,
                   "SELECT pickle FROM main.Matches"
                   " WHERE id IN (SELECT id FROM main.Matches "
                   "WHERE cls = ? AND firstPlayer = ? and secondPlayer = ? "
                   "ORDER BY RANDOM() LIMIT ?)",
                   (cls, playerOne, playerTwo, n,))

    @staticmethod
    def __simple_sample(cursor: sqlite3.Cursor, n: int):
        return _qw(cursor,
                   "SELECT pickle FROM main.Matches"
                   " WHERE id IN (SELECT id FROM main.Matches ORDER BY RANDOM() LIMIT ?)",
                   (n,))


if __name__ == "__main__":
    matches = MatchHistory.sample(10, cls="MCTS_VS_MCTS")
    for match in matches:
        for board in match.boards:
            board.print()
