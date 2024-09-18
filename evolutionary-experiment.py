import matplotlib.pyplot as plt
from time import time
from macht.macht.term import main
from statistics import mean

def evolutionary_experiment(agent_class, generations, num_agents, num_games, agents_that_survive, scale, dynamic_viz=False):
    #  initialize first generation of agents
    agents = [agent_class for _ in range(num_agents)]
    history_of_max_scores = []
    start = time()
    for generation_num in range(generations):
        # evaluates agents
        max_avg_score = -1
        for agent in agents:
            avg_score = 0
            for game in range(num_games):
                final_score, best_tile = main(args=['--auto'], agent=agent)
                avg_score += final_score/num_games
            agent.performance = avg_score
            if avg_score > max_avg_score: max_avg_score = avg_score

        history_of_max_scores.append(max_avg_score)
        running_avg = mean(history_of_max_scores)
        if dynamic_viz:
            plt.scatter(generation_num, max_avg_score, c='red')
            plt.scatter(generation_num, running_avg, c='orange')
            plt.title(f'2048 {agent.name} Score')
            plt.xlabel(f'Generation Number (Completed in {(time()-start):.2f}s)')
            plt.ylabel(f'Average Score of Best Agent ({running_avg:.1f})')
            plt.pause(0.00001)

        # pick best agents to continue
        agent_index_buffer = np.argsort(np.array([agent.performance for agent in agents]))[(num_agents - agents_that_survive):]
        agent_buffer = [agents[i] for i in agent_index_buffer]
        # if dynamic_viz: plt.show()

        # add new agents for next gen
        agents = agent_buffer + [EvolutionaryAgent() for _ in range(num_agents - agents_that_survive)]

        # mutate
        for agent in agents:
            agent.mutate(scale=scale)
    if dynamic_viz: plt.show()
    print(f'Completed {num_trials} in {(time()-start):.5}s')

if __name__ == '__main__':
    evolutionary_experiment(
        agent_class=EvolutionaryAgent(), \
        generations=1000, \
        num_agents=2, \
        num_games=20, \
        agents_that_survive=1, \
        scale =0.05, \
        dynamic_viz=True
    )
