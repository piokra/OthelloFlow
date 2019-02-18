from evaluator import ZeroEvaluator
from match import Match
from mcts import MonteCarloTS
from model_evaluator import ModelEvaluator
from strategy_factory import StrategyFactory

if __name__ == "__main__":
    modelFactory = StrategyFactory(MonteCarloTS, playerOneEvaluator=ModelEvaluator(1, "../models/6x6eval.tf"),
                                   playerTwoEvaluator=ModelEvaluator(-1, "../models/6x6eval.tf"),
                                   maxSeconds=3, alpha=1)
    mctsFactory = StrategyFactory(MonteCarloTS, playerOneEvaluator=ZeroEvaluator(),
                                  playerTwoEvaluator=ZeroEvaluator(),
                                  maxSeconds=3, alpha=1)

    won, lost, drawn = 0, 0, 0
    for _ in range(10):
        match = Match(modelFactory, mctsFactory)
        match.play()
        one, two = match.playerOne.state.score()
        if one > two:
            won += 1
        elif two > one:
            lost += 1
        else:
            drawn += 1
        match.save("MODEL_VS_MCTS")
        print(".", end="")
    print(won, lost, drawn)
