# Playing 2048 with RL and EC

This project includes a python clone of 2048, along with various custom **Numpy-only** reinforcement learning agents to play the game. It also includes a evolutionary computation approach using the [PyTorch NEAT](https://github.com/uber-research/PyTorch-NEAT) repository.
Finally, it includes a deep learining approach.

<!-- ![](.assets/viz.gif) -->

In reinforcement learning terms, 2048 is an episodic game with dense rewards, in which the actions are swiping up, down, left, and right, the states are the boards right after the swipe, and the afterstates are the board after a new tile spawns in.

## Repo Guide

```
2048rl/
├── 2048/           // modification of original source code to view game
├── assets/         // various visuals and movies created
├── game/           // custom implementation of game
├── scratch/        // from scratch implementation of TD learning with N-tuple nets
├── benchmark.py    // 
├── evaluate.py     // evaluate agent
├── play.py         // play custom 2048 game via terminal
├── README.md
└── workspace.ipynb // working environment for development
```

## Resources:

Resources for DeepRL approach:
- TODO

Resources used for from scratch TD Learning: 
- [Temporal Difference Learning of N-Tuple Networks for the Game 2048](https://www.cs.put.poznan.pl/wjaskowski/pub/papers/Szubert2014_2048.pdf) for a 
- [Python implementation of above paper](https://github.com/alanhyue/RL-2048-with-n-tuple-network) for sanity checks.
- [On Reinforcement Learning for the Game of 2048](https://arxiv.org/pdf/2212.11087) for an overview of RL techniques used to play 2048. 
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) by Sutton and Barto, for a comprehensive introdution to the field. 
- [The original 2048 repository](https://github.com/gabrielecirulli/2048) for the visualization and the original popularization of the game. 
- [The macht repository](https://github.com/rolfmorel/macht) for testing agents before I implemented the clone.

## Project TODOs
- Implement and test [expectimax](https://en.wikipedia.org/wiki/Expectiminimax) n-ply search, currently doing just greedy/1-ply 
- Evaluate raw policy and policy with search
- [This](https://2048verse.com/) is site for best players, 1M+ games.
