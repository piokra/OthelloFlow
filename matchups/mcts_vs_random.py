from evaluator import ZeroEvaluator
from match import Match
from mcts import MonteCarloTS
from random_strategy import RandomStrategy
from strategy_factory import StrategyFactory

if __name__ == "__main__":
    mctsFactory = StrategyFactory(MonteCarloTS, playerOneEvaluator=ZeroEvaluator(), playerTwoEvaluator=ZeroEvaluator(),
                                  maxSeconds=2)
    randomFactory = StrategyFactory(RandomStrategy)

    won, lost, drawn = 0, 0, 0
    for _ in range(100):
        match = Match(mctsFactory, randomFactory)
        match.play()
        one, two = match.playerOne.state.score()
        if one > two:
            won += 1
        elif two > one:
            lost += 1
        else:
            drawn += 1

    print(won, lost, drawn)
