# Reinforcement Learning for 2048
A custom python clone of 2048, along with various custom reinforcement learning agents to play the game.

In reinforcement learning terms, 2048 is an episodic game with dense rewards, in which the actions are swiping up, down, left, and right, the states are the boards right after the swipe, and the afterstates are the board after a new tile spawns in.

The statistics tracked for a specific agents are listed below. Each statistic is an average over x games
- Performance Stats:
    - Average score achieved
    - Highest score achieved
    - Average highest tile
    - Highest tile achieved
    - Percentage of games that reach 2048 tile
- Training Stats:
    - Average number of games to first 2048 tile (in training!)

The agents that resulted in significant performance are listed below.
- A TD(0) Agent using an n-tuple network to evaluate aterstates
    - Tuple design= TupleMap1 in `functions/tuplefuncs.py`, Learning rate=, achieves:

20.2% of games reached 2048, 1.5% reached 4196. The highest score was 68840. 100,000 games in 6.1 hrs
49.1% of gamer reach 2048, 1.5% reached 4196. The highest score war 56208. 103,00 games in 11 hours. 92 MB of params

## Resources:
The various resources used for this project and how they were used are listed below. 
- [Temporal Difference Learning of N-Tuple Networks for the Game 2048](https://www.cs.put.poznan.pl/wjaskowski/pub/papers/Szubert2014_2048.pdf) for a 
- [Python implementation of above paper](https://github.com/alanhyue/RL-2048-with-n-tuple-network) for sanity checks.
- [On Reinforcement Learning for the Game of 2048](https://arxiv.org/pdf/2212.11087) for an overview of RL techniques used to play 2048. 
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) by Sutton and Barto, for a comprehensive introdution to the field. 
- [The original 2048 repository](https://github.com/gabrielecirulli/2048) for the visualization and the original popularization of the game. 
- [The macht repository](https://github.com/rolfmorel/macht) for testing agents before I implemented a custom game.

## Project TODOs
- Implement "watch agent play" feature
- Implement grid search for various params
- Saved models directory with version number and hyperparams recording, add this to gitignore
- Num parameters feature in ntuplenet
- Make stats section of README a table
- Make visual for 2 or 3 games side by side with caption, hook up finished agent to [original repo](https://github.com/gabrielecirulli/2048)
- Linkedin Post with explan
    I wrote a reinforcement learning agent from scratch to play 2048. Instead of telling an algorithm how to win the game, an RL agent is just told to get a high score. It learns to achive a high score through playing many games. 

    While playing, the agent achieves the 2048 tile for the first time during its x-th game. After playing y games, the highest tile is achieves is z, and achives this in about z% of games. 

    This is neither the best way to implement a 2048 bot, nor the best use case for RL. I chose this project as an academic exersize and because I like the game. Implementation and performance details, as well as the source code, can be found on my GitHub.
- Small YT vid

## Structure of Repo
- The `game` directory stores the `gameof2048.py` file that implements the game. 
- The `agents` directory stores custom agent implementations, and are partially composed of classes from the `models` and `functions` directories.
- The `experiments` directory stores experiments for different types of agents

## Notes
- To achieve tile x, you need at minimum to have achieved x(log_2(x)-2) points, MUCH more likely to be around x(log_2(x)-1). This means only games of 20k+ points could have achieved 2048, etc 
