import logging
import torch

import neat.population as pop
import neat.experiments.gameof2048.config as c
from neat.phenotype.feed_forward import FeedForwardNet
from neat.experiments.gameof2048.gameof2048 import Gameof2048

from neat.visualize import draw_net

logger = logging.getLogger(__name__)

logger.info(f'Beginning Experiment on device: {c.Gameof2048Config.DEVICE}')
neat = pop.Population(c.Gameof2048Config)
solution, generation = neat.run()

if solution is not None:
    logger.info('Evaluating solution...')

    solution_phenotype = FeedForwardNet(solution, c.Gameof2048Config)

    agent = c.NEATAgent(phenotype=solution_phenotype, device=c.Gameof2048Config.DEVICE)

    score_history = []
    for game in range(c.Gameof2048Config.NUM_TEST_GAMES):
        game = Gameof2048(agent=agent)
        final_score = game.play()
        score_history.append(final_score)

    logger.info(f'After {c.Gameof2048Config.NUMBER_OF_GENERATIONS} Generations, Mean Score: {sum(score_history)/len(score_history)} Max Score: {max(score_history)}')

    # TODO view solution 
    draw_net(solution, view=True, filename='./images/2048-solution', show_disabled=True)
