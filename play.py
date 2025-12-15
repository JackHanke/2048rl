from scratch.agents.humanagent import HumanAgent
from game.gameof2048 import Gameof2048

game = Gameof2048(agent=HumanAgent())
game.play()