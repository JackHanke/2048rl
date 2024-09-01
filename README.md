# Reinforcement Learning for 2048 Game

This is an unofficial fork of the [Macht repo](https://github.com/rolfmorel/macht), a Python clone of 2048, for learning various Reinforcement Learning techniques. Another resource was [this comprehensive review](https://arxiv.org/pdf/2212.11087) of the current state of 2048 RL Work as of 2023. 

## Structure of Repo
The `agents` directory stores custom agent implementations, and are partially composed of classes from the `models` and `functions` directories.
The `experiments` directory stores experiments for different types of agents

## TODO
- Classify the problem:
    - The states are the board states. States are transformed into different representations with functions in `functions/rlfuncs.py`.
    - The actions are the legal moves in the position (left,right,up,down). Use invalid action masking for policy gradient methods!
    - The reward for an action is the score delta obtained from a specific move
    - The return is the predicted sum of future rewards. This is the predicted total score. Is discounting appropriate? Should the reward be normalized?
    - Episodic, the game eventually ends

- Code:
    - Implement grid search for various params
    - Implement agent save feature 

- Make visual for 2 or 3 games side by side with caption, hook up finished agent to [original repo](https://github.com/gabrielecirulli/2048)

- Statistics I would like to track for specific agents are:
    - Performance Stats:
        - Average score achieved
        - Highest score achieved
        - Average highest tile
        - Highest tile achieved
        - Percentage of games that reach 2048 tile
    - Training Stats:
        - Average number of games to first 2048 tile (in training!)

- Post with explan
    This is a video of a custom reinforcement learning agent I wrote playing 2048. Instead of telling the algorithm how to win the game, you just tell it to get a high score. It plays the game and learns what behaviors maximizes its chance of obtaining a high score.

    After x games, the agent achieves the 2048 tile for the first time. The highest tile is achieves is , and achives this in about z% of games. 
    After y games, the agent achieved an average high score, and an all time best score of 

    This is neither the best way to implement a 2048 bot, nor the best use case for deep RL. I chose this project as an academic exersize while reading Reinforcement Learning: An Introduction by Sutton and Barto, and because I like the game. The code can be found at my repo. 



