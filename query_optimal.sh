#!/bin/bash

# need the data file in the current directory, so switch to the directory
# where it is
ASSIGNMENT_PATH="/c/cs474/hw6"
cd "$ASSIGNMENT_PATH"

java -classpath yahtzee.jar com.bloxomo.gametheory.yahtzee.StrategyQuery com.bloxomo.gametheory.yahtzee.BonusYahtzeeState com.bloxomo.gametheory.yahtzee.OptimalStrategy
