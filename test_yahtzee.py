import sys
import yahtzee
import nn

def main():
    if len(sys.argv) == 1:
        print("USAGE: python3", sys.argv[0], " num-games")
        sys.exit(1)

    try:
        games = int(sys.argv[1])
        if games <= 0:
            raise ValueError("number of games must be positive")
    except ValueError as ex:
        print("USAGE: python3", sys.argv[0], " num-games")
        sys.exit(1)
        
    model = nn.train();
    strat = nn.NNStrategy(model)
    print("RESULT:", yahtzee.evaluate_strategy(games, strat.choose_dice, strat.choose_category))

    
if __name__ == "__main__":
    main()
    
