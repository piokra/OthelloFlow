import numpy as np
import sqlite3
import pickle
from typing import Tuple

from match import Match

def sample_mcts_data(n, last, path, size=6):
    conn = sqlite3.connect(path)
    curr = conn.cursor()
    max, = curr.execute("SELECT MAX(game_nb) FROM MCTSData").fetchone()
    print(max)
    if max is None:
        max = 0
    curr = curr.execute("SELECT pickle FROM main.MCTSData"
                   " WHERE id IN (SELECT id FROM main.MCTSData "
                   "WHERE game_nb >= ?"
                   "ORDER BY RANDOM() LIMIT ?)", (max-last, n))

    ret = curr.fetchall()
    boards, vals, policies = [], [], []
    curr.close()
    conn.close()

    for (pickles,) in ret:
        board, val, policy = pickle.loads(pickles)
        boards.append(board)
        vals.append(val)
        policies.append(policy)

    return np.array(boards).reshape((len(boards), size, size, 1)), \
           np.array(vals).reshape((len(vals),)), \
           np.array(policies).reshape((len(policies), size, size)),



if __name__ == "__main__":
    board, val, policy = sample_mcts_data(20, 50, "matches.db")
    print(board)