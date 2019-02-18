import pickle
import sqlite3

import tensorflow as tf
import numpy as np

from board import Board
from strategy import Strategy

from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libc.math cimport exp
from tensorflow.python.keras.models import load_model, Model

DEF MAX_BOARD_SIZE = 6
DEF VERBOSE = 0
cdef short DIRSX[8]
cdef short DIRSY[8]
cdef short ZERO_BOARD[MAX_BOARD_SIZE][MAX_BOARD_SIZE]
ZERO_BOARD = [[0] * MAX_BOARD_SIZE] * MAX_BOARD_SIZE
DIRSX[:] = [1, 0, -1, 0, 1, 1, -1, -1]
DIRSY[:] = [0, 1, 0, -1, 1, -1, 1, -1]

cdef struct CMove:
    short player, x, y

cdef struct CBoard:
    int size
    short board[MAX_BOARD_SIZE][MAX_BOARD_SIZE]

cdef struct CValidDirections:
    short size
    short dirsx[8]
    short dirsy[8]

cdef struct CEdge:
    int visits
    double prob, value
    CLeaf* leaf

cdef CEdge* CEdgePtr

cdef struct CLeaf:
    CBoard board
    CEdge* edges[MAX_BOARD_SIZE][MAX_BOARD_SIZE]

cdef struct CValidMoves:
    int size
    short movex[MAX_BOARD_SIZE*MAX_BOARD_SIZE]
    short movey[MAX_BOARD_SIZE*MAX_BOARD_SIZE]

cdef short board_state(CBoard*board, CMove move):
    return board.board[move.x][move.y]

cdef CBoard*set_board_state(CBoard*board, CMove move):
    board.board[move.x][move.y] = move.player
    return board

cdef CBoard init_board(int size):
    cdef CBoard board
    board.board = ZERO_BOARD
    board.size = size
    board.board[size / 2][size / 2] = 1
    board.board[size / 2 - 1][size / 2 - 1] = 1
    board.board[size / 2 - 1][size / 2] = -1
    board.board[size / 2][size / 2 - 1] = -1
    return board

cdef CBoard*flip_board(CBoard*board):
    for i in range(board.size):
        for j in range(board.size):
            board.board[i][j] = -board.board[i][j]
    return board

cdef short is_move_valid(CBoard*board, CMove move):
    for i in range(8):
        if _is_direction_valid(board, move, DIRSX[i], DIRSY[i]):
            return 1
    return 0

cdef CValidDirections valid_directions(CBoard*board, CMove move):
    cdef CValidDirections ret
    cdef int valid = 0
    for i in range(8):
        if _is_direction_valid(board, move, DIRSX[i], DIRSY[i]):
            ret.dirsx[valid] = DIRSX[i]
            ret.dirsy[valid] = DIRSY[i]
            valid += 1
    ret.size = valid

    return ret

cdef short _is_direction_valid(CBoard*board, CMove move, int dx, int dy):
    cdef CMove test_move = move
    cdef short met_opponent = 0
    test_move.x += dx
    test_move.y += dy
    if board_state(board, move) != 0:
        return 0

    while _is_in_bounds(board, test_move):
        if board_state(board, test_move) == 0:
            return 0

        if board_state(board, test_move) == move.player:
            if met_opponent:
                return 1
            else:
                return 0

        met_opponent = 1
        test_move.x += dx
        test_move.y += dy
    return 0

cdef short _is_in_bounds(CBoard*board, CMove move):
    if move.x < 0 or move.y < 0:
        return 0

    if move.x >= board.size or move.y >= board.size:
        return 0

    return 1

cdef CBoard*make_a_move(CBoard*board, CMove move):
    cdef CMove test_move = move
    cdef CValidDirections dirs = valid_directions(board, move)
    cdef (short, short) direction
    for i in range(dirs.size):
        direction = (dirs.dirsx[i], dirs.dirsy[i])
        test_move = move
        test_move.x += direction[0]
        test_move.y += direction[1]
        while board_state(board, test_move) != move.player:
            set_board_state(board, test_move)
            test_move.x += direction[0]
            test_move.y += direction[1]

    set_board_state(board, move)
    return board

cdef print_board(CBoard*board):
    cdef int e, b = 0, w = 0
    for i in range(MAX_BOARD_SIZE):
        for j in range(MAX_BOARD_SIZE):
            e = board.board[i][j]
            if e == 1:
                b += 1
            if e == -1:
                w += 1
            printf("%2d ", e)
        printf('\n')
    print(b, w, b+w)

cdef short has_valid_moves(CBoard*board, short player):
    for i in range(board.size):
        for j in range(board.size):
            if is_move_valid(board, CMove(player, i, j)):
                return 1
    return 0

cdef CValidMoves valid_moves(CBoard*board, short player):
    cdef int counter = 0
    cdef CValidMoves ret
    for i in range(board.size):
        for j in range(board.size):
            if is_move_valid(board, CMove(player, i, j)):
                ret.movex[counter] = i
                ret.movey[counter] = j
                counter += 1
    ret.size = counter
    return ret

cdef CLeaf* init_tree(int size):
    cdef CLeaf* ret = <CLeaf*> malloc(sizeof(CLeaf))
    ret.board = init_board(size)
    for i in range(size):
        for j in range(size):
            ret.edges[i][j] = NULL
    return ret

cdef CEdge* init_edge(CLeaf* start, CMove move, double prob):
    cdef CEdge* ret = <CEdge*> malloc(sizeof(CEdge))
    ret.prob = prob
    ret.leaf = init_tree(start.board.size)
    ret.leaf.board = start.board
    make_a_move(&ret.leaf.board, move)
    ret.value = 0
    ret.visits = 0
    return ret


cdef void _free_tree(CLeaf* tree):
    for i in range(tree.board.size):
        for j in range(tree.board.size):
            if tree.edges[i][j] != NULL:
                _free_tree(tree.edges[i][j].leaf)
    free(tree)

cdef  (CLeaf*, int, int) _choose_a_move(CLeaf* tree, int player, int size, value_model, policy_model, int max_depth, int iterations):
    cdef CMove ret
    cdef CLeaf* current = tree
    cdef CValidMoves _valid_moves
    cdef int depth = 0, current_player = player 
    cdef double[:,:] policy_view
    cdef double[:] value_view
    cdef double value, max_mean
    cdef int x, y, tot_n = 0, n, max_i, max_visits
    cdef int first_visit = 0
    cdef double local_prob
    cdef double candidates[MAX_BOARD_SIZE*MAX_BOARD_SIZE]
    cdef CMove history[MAX_BOARD_SIZE*MAX_BOARD_SIZE]

    if tree == NULL:
        tree = init_tree(size)
        current = tree

    for _ in range(iterations):
        depth = 0
        current_player = player
        current = tree
        while depth <= max_depth and (has_valid_moves(&(current.board), 1) or has_valid_moves(&current.board, -1)):
            _valid_moves = valid_moves(&current.board, current_player)
            if _valid_moves.size == 0:
                current_player *= -1
                continue

            first_visit = 1
            for _x in range(size):
                for _y in range(size):
                    if current.edges[_x][_y]:
                        first_visit = 0

            if first_visit:
                if current_player == -1:
                    flip_board(&current.board)
                policy_view = policy_model.predict(np.asarray(current.board.board)[:size,:size].reshape((1,size,size,1))).reshape(size, size).astype('float64')
                if current_player == -1:
                    flip_board(&current.board)
            tot_n = 0
            for i in range(_valid_moves.size):
                x = _valid_moves.movex[i]
                y = _valid_moves.movey[i]
                if first_visit:
                    local_prob = policy_view[x, y]
                    current.edges[x][y] = init_edge(current, CMove(current_player, x, y), local_prob)
                else:
                    local_prob = current.edges[x][y].prob
                    tot_n += current.edges[x][y].visits
                candidates[i] = local_prob
            max_mean = -100
            for i in range(_valid_moves.size):
                x = _valid_moves.movex[i]
                y = _valid_moves.movey[i]
                value = current.edges[x][y].value
                n = current.edges[x][y].visits
                if n:
                    candidates[i] = candidates[i]*exp(-10*(n/tot_n)) + value/n*(1-exp(-10*(n/tot_n)))

            np_probs = np.asarray(candidates)
            np_probs = np_probs[:_valid_moves.size]
            np_probs[np_probs < 0] = 0
            np_probs /= np.sum(np_probs)

            max_i = np.random.choice(range(_valid_moves.size), p=np_probs)
            max_mean = candidates[max_i]
            x = _valid_moves.movex[max_i]
            y = _valid_moves.movey[max_i]
            history[depth] = CMove(current_player, x, y)
            current = current.edges[x][y].leaf
            current_player *= -1
            depth += 1

        if current_player == -1:
            flip_board(&current.board)
        value_view = value_model.predict(np.asarray(current.board.board)[:size,:size].reshape((1,size,size,1))).reshape((1,)).astype('float64')
        if current_player == -1:
            flip_board(&current.board)
        value = value_view[0]
        current = tree
        for i in range(depth):
            x, y = history[i].x, history[i].y

            current.edges[x][y].visits += 1
            current.edges[x][y].value += value
            current = current.edges[x][y].leaf

    max_x, max_y = -1, -1
    max_visits = 0
    for x in range(MAX_BOARD_SIZE):
        for y in range(MAX_BOARD_SIZE):

            if tree.edges[x][y] and tree.edges[x][y].visits > max_visits:
                max_visits = tree.edges[x][y].visits
                max_x, max_y = x, y
            if VERBOSE:
                if tree.edges[x][y]:
                    printf("%10d %2d ", tree.edges[x][y].visits, tree.board.board[x][y])
                else:
                    printf("%10d %2d", 0, tree.board.board[x][y])
        if VERBOSE:
            print()

    return tree.edges[max_x][max_y].leaf, max_x, max_y


def __():
    pass

cdef class OptMCTS:
    cdef CLeaf* root
    cdef CLeaf* current
    cdef int player, max_depth, iterations, size, move, won
    cdef object value_model, policy_model
    cdef list history
    cdef public object state

    def __init__(self, player, size, value_model, policy_model, max_depth = 10, iterations = 160, **kwargs):
        # CLeaf* tree, int player, int size, value_model, policy_model, int max_depth, int iterations
        self.root = init_tree(size)
        self.current = self.root

        self.value_model = value_model
        self.policy_model = policy_model
        self.max_depth = max_depth
        self.iterations = iterations
        self.player = player
        self.history = []
        self.state = Board(**kwargs)
        self.size = size
        self.move = 0

    def save(self, path='matches.db'):
        cdef CLeaf* current
        cdef CEdge* edge
        conn = sqlite3.connect(path)
        curr = conn.cursor()
        curr.execute("SELECT max(game_nb) from MCTSData")
        game_nb,  = curr.fetchone()
        if game_nb is None:
            game_nb = 0

        current = self.root
        for player, x, y in self.history:
            current = current.edges[x][y].leaf


        board = np.asarray(init_board(self.size).board)
        value = np.sign(np.sum(np.asarray(current.board.board)))
        current = self.root

        for player, x, y in self.history:
            policy = np.zeros((self.size, self.size), dtype='float64')
            n = 1
            for i in range(self.size):
                for j in range(self.size):
                    n += current.edges[x][y].visits
                    policy[i,j] = current.edges[x][y].visits

            policy /= n
            edge = current.edges[x][y]

            if player == -1:
                flip_board(&current.board)
            board = np.asarray(current.board.board)

            if player == -1:
                flip_board(&current.board)


            curr.execute("INSERT INTO MCTSData(game_nb, pickle) VALUES(?, ?)",
                         (game_nb, pickle.dumps((board, value, policy))))

            # value = edge.value/edge.visits
            current = edge.leaf

        curr.close()
        conn.commit()
        conn.close()

    def free(self):
        _free_tree(self.root)

    def return_a_move(self):
        cdef CLeaf* discard
        cdef int x, y
        discard, x, y = _choose_a_move(self.current, self.player, self.size, self.value_model, self.policy_model, self.max_depth, self.iterations)
        return x, y

    def make_a_move(self):
        self.move += 1
        self.current, x, y = _choose_a_move(self.current, self.player, self.size, self.value_model, self.policy_model, self.max_depth, self.iterations)
        self.state.make_a_move(x, y, self.player)
        self.history.append((self.player, x, y))
        if VERBOSE:
            print(self.move)
            print_board(&self.current.board)
        return x, y

    def opponent_move(self, x, y):
        self.move += 1
        self.state.make_a_move(x, y, -self.player)
        self.history.append((-self.player, x, y))
        if self.current.edges[x][y] == NULL:
            self.current.edges[x][y] = init_edge(self.current, CMove(-self.player, x, y), 1)
        self.current = self.current.edges[x][y].leaf

    cdef CLeaf* _current(self):
        return self.current

cdef class CorrOptMCTS(OptMCTS):
    def __init__(self, OptMCTS other, player, size, value_model, policy_model, max_depth = 10, iterations = 160, **kwargs):
        super().__init__(player, size, value_model, policy_model, max_depth, iterations, **kwargs)
        self.root = other.root
        self.current = other.current

    def free(self):
        pass

cdef class OPTFactory:
    cdef OptMCTS player_one_strategy
    cdef OptMCTS player_two_strategy
    cdef int size, max_depth, iterations
    cdef object value_model, policy_model
    def __init__(self, size, value_model, policy_model, max_depth = 10, iterations = 160):
        self.size = size
        self.max_depth = max_depth
        self.iterations = iterations
        self.value_model = value_model
        self.policy_model = policy_model
        self.player_one_strategy = None
        self.player_two_strategy = None
        # self.player_one_strategy = OptMCTS(1, size, value_model, policy_model, max_depth, iterations)
        # self.player_two_strategy = CorrOptMCTS(self.player_one_strategy, -1, size, value_model, policy_model, max_depth, iterations)
    def build(self, player: int, board):
        if player == 1:
            if self.player_one_strategy is not None:
                self.player_one_strategy.free()
            self.player_one_strategy = OptMCTS(1, self.size, self.value_model,
                                               self.policy_model, self.max_depth, self.iterations)
            self.player_two_strategy = CorrOptMCTS(self.player_one_strategy, -1,
                                               self.size, self.value_model, self.policy_model, self.max_depth, self.iterations)
            return self.player_one_strategy
        else:
            if self.player_one_strategy is None:
                if self.player_two_strategy is not None:
                    self.player_two_strategy.free()
                self.player_two_strategy = OptMCTS(-1,
                                               self.size, self.value_model, self.policy_model, self.max_depth, self.iterations)
            return self.player_two_strategy

def test():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    _choose_a_move(NULL, 1, 6, load_model("../models/alphazerolike/azval.tf"), load_model("../models/alphazerolike/azpol.tf"), 10, 160)
