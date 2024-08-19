from macht.macht.term import main
from statistics import stdev, median, mean

# baseline
baseline = False
if baseline:
    scores = []
    num_trials = 1000
    for _ in range(num_trials):
        val = main(args=['--auto','True'])
        scores.append(val)
    print(sum(scores)/len(scores))
    print(f'mean(scores) for {num_trials} trials is {mean(scores)}')
    # mean ~ 1096
    print(f'stdev(scores) for {num_trials} trials is {stdev(scores)}')
    # stdev ~ 549
    print(f'median(scores) for {num_trials} trials is {median(scores)}')
    # median ~ 1048


class QLearningAgent:
    def __init__(self):
        pass

    def update(): #
        pass

    def choose(): #
        pass

class ActorCriticAgent:
    pass
