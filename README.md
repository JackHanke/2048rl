# Reinforcement Learning for 2048 Game

- Actually do the RL
    - The states are the board states
    - The actions are the legal moves in the position (left,right,up,down)
    - The reward for an action is the score delta obtained from a specific move
    - The return is the predicted sum of future rewards. This is the predicted total score
    - Is discounting appropriate? probably not
    - Episodic, the game eventually ends

- Hook up finished agent to [original repo](https://github.com/gabrielecirulli/2048)
- Make visual
- Read about other [work](https://en.wikipedia.org/wiki/2048_(video_game))
- Post with explan
    This is a video of a custom deepRL agent I wrote playing 2048. Instead of telling the algorithm how to win the game, you just tell it to get a high score. It plays the game and learns what behaviors maximizes it's chance of obtaining a high score.

    After x hours, the system achieves the 2048 tile for the first time
    After y hours, the system achieves the 4096 tile for the first time
    After z hours, the system achieves the w tile x% of the time

    This is neither the best way to implement a 2048 bot, nor the best use case for deep RL. I chose this project just as an academic excersize while I read thorugh Reinforcement Learning: An Introduction by Sutton and Barto, and because I like the game. The code can be found at my repo. 



