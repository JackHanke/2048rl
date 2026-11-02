import logging

import neat.population as pop
import neat.experiments.ttt.config as c
from neat.phenotype.feed_forward import FeedForwardNet

from neat.visualize import draw_net

logger = logging.getLogger(__name__)

logger.info(f'Beginning Experiment on device: {c.tttConfig.DEVICE}')

neat = pop.Population(c.tttConfig)
solution, generation = neat.run()

if solution is not None:
    logger.info('Evaluating solution...')

    solution_phenotype = FeedForwardNet(solution, c.tttConfig)

    logger.info(f'After {c.tttConfig.NUMBER_OF_GENERATIONS} Generations')

    # TODO view solution 
    draw_net(solution, view=True, filename='./images/2048-solution', show_disabled=True)
