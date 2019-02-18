from evaluator import ZeroEvaluator
from match import Match
from mcts import MonteCarloTS
from strategy_factory import StrategyFactory

if __name__ == "__main__":
    mctsFactory = StrategyFactory(MonteCarloTS, playerOneEvaluator=ZeroEvaluator(), playerTwoEvaluator=ZeroEvaluator(),
                                  maxSeconds=3)
    mctsFactoryTwo = StrategyFactory(MonteCarloTS, playerOneEvaluator=ZeroEvaluator(),
                                     playerTwoEvaluator=ZeroEvaluator(),
                                     maxSeconds=3)

    won, lost, drawn = 0, 0, 0
    for _ in range(1000):
        match = Match(mctsFactory, mctsFactoryTwo)
        match.play()
        one, two = match.playerOne.state.score()
        if one > two:
            won += 1
        elif two > one:
            lost += 1
        else:
            drawn += 1
        match.save("MCTS_VS_MCTS")
        print(".", end="")
    print(won, lost, drawn)
