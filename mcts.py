import numpy as np
from typing import Tuple, Set, Dict

from board import Board
from evaluator import ZeroEvaluator
from strategy import Strategy
from time import time


class MonteCarloTS(Strategy):
    def __init__(self, player, board: Board, playerOneEvaluator, playerTwoEvaluator, maxDepth=24, maxSeconds=10,
                 epsilon=0.1, alpha=0.3):
        super().__init__(player, board)
        self.evaluators = {1: playerOneEvaluator, -1: playerTwoEvaluator}
        self.maxDepth = maxDepth
        self.maxSeconds = maxSeconds
        self.epsilon = epsilon
        self.visited = set()
        self.alpha = alpha

    def return_a_move(self) -> Tuple[int, int]:
        explored_moves = self.state.valid_moves(self.player)
        if len(explored_moves) == 1:
            return explored_moves[0]

        if len(explored_moves) == 0:
            return -1, -1

        explored_states = []
        for move in explored_moves:
            new_state = self.state.clone()
            new_state.make_a_move(*move, self.player)
            explored_states.append(new_state)

        tree_state = dict()
        now = time()
        player = self.player
        tree_state[self.state.clone()] = (0, 0)

        while time() < now + self.maxSeconds:
            done = False
            depth = 0
            current = self.state.clone()
            history = set()

            while depth <= self.maxDepth:
                move = self.__choose_a_branch(tree_state, current, player)

                if move is None:
                    break

                current.make_a_move(*move, player)
                if current not in tree_state:
                    tree_state[current.clone()] = (0, 0)
                history.add(current.clone())

                depth += 1
                if current.has_valid_move(-player):
                    player = -player
            history.add(current)

            one, two = current.score()
            won = 0
            if (one - two) * self.player >= 0:
                won += 1
            for state in history:
                results = tree_state[state]
                results = (results[0] + won, results[1] + 1)
                tree_state[state] = results

        best = explored_moves[0]
        best_value = -1
        for move, end in zip(explored_moves, explored_states):
            value = 0
            if end in tree_state:
                score = tree_state[end]
                value = 0
                if score[1]:
                    value = score[0] / score[1]

            if best_value < value:
                best_value = value
                best = move

        return best

    def __choose_a_branch(self, tree_state: Dict[Board, Tuple[int, int]], state: Board, player):
        valid_moves = state.valid_moves(player)
        if not len(valid_moves):
            return None
        valid_diff = []
        for move in valid_moves:
            clone = state.clone()
            clone.make_a_move(*move, player)
            valid_diff.append((move, clone))
        probabilities = list(map(lambda me: self.__evaluate(tree_state, player, state, *me), valid_diff))
        s = sum(probabilities)
        probabilities = list(map(lambda x: x / s, probabilities))
        index = np.random.choice(range(len(probabilities)), p=probabilities)
        return valid_moves[index]

    def __evaluate(self, tree_state: Dict[Board, Tuple[int, int]], player, move: Tuple[int, int], start: Board,
                   end: Board):
        e = 0
        if end in tree_state:
            wl = tree_state[end]
            if self.player != player:
                wl = (wl[1] - wl[0], wl[1])
            if wl[1] > 0:
                e = wl[0] / wl[1]

        if e:
            return self.epsilon + (1 - self.alpha) * e + self.alpha * self.evaluators[player].evaluate(
                start, move, end)
        else:
            return self.epsilon + self.evaluators[player].evaluate(start, move, end)


if __name__ == "__main__":
    from random_strategy import RandomStrategy
    from strategy_factory import StrategyFactory
    from match import Match

    board = Board()
    mcts = MonteCarloTS(1, board, ZeroEvaluator(), ZeroEvaluator())

    mctsStrategyFactory = StrategyFactory(MonteCarloTS, playerOneEvaluator=ZeroEvaluator(),
                                          playerTwoEvaluator=ZeroEvaluator(), maxSeconds=1)
    randomFactory = StrategyFactory(RandomStrategy)

    match = Match(mctsStrategyFactory, randomFactory)
    match.play()
    print(match.playerOne.state.score())
    match.playerOne.state.print()
