import matplotlib.pyplot as plt
from time import time
from macht.macht.term import main
from statistics import mean

# mean ~ 1096 points
# stdev ~ 549 points
# median ~ 1048 points
def baseline(agent_class, num_trials):
    scores = []
    start = time()
    for trial_num in range(num_trials):
        final_score, best_tile = main(args=['--auto'], agent=agent_class)
        scores.append(final_score)
    print(f'Data collected in {(time()-start):.2f}s')
    plt.hist(scores, bins=[25*i for i in range(150)], color='red', density=True)
    plt.title(f'2048 {agent_class.name} Score')
    plt.xlabel(f'Trial Number (Completed in {(time()-start):.2f}s)')
    plt.ylabel(f'Final Score, Running Average = {running_avg:.1f} Points')
    plt.show()

if __name__ == '__main__':
    baseline(agent_class=DumbAgent(), num_trials=20000)
