import matplotlib.pyplot as plt
# TODO rewrite 2048 env in gymnasium format for easy swap of envs for a specific agent

# import gymnasium as gym
# env = gym.make("LunarLander-v2", render_mode="human")
# observation, info = env.reset()
# for _ in range(1000):
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()

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

def offline_experiment(agent, num_trials, dynamic_viz=False):
    scores = []
    weights = []
    grads = []
    start = time()
    for trial_num in range(num_trials):
        # final_score, best_tile = main(args=['--auto', '--viz'], agent=agent)
        try:
            final_score, best_tile = main(args=['--auto'], agent=agent)
        except Exception as e:
            input(e)
        scores.append(final_score)
        running_avg = mean(scores)

        if dynamic_viz:
            # plt.subplot(1,2,1)
            plt.scatter(trial_num, final_score, c='red')
            plt.scatter(trial_num, running_avg, c='orange')
            plt.title(f'2048 {agent.name} Score')
            plt.xlabel(f'Trial Number (Completed in {(time()-start):.2f}s)')
            plt.ylabel(f'Final Score, Running Average = {running_avg:.1f} Points')

            plt.pause(0.00001)
        else:
            # print(f'Running average = {running_avg:.1f} \r\033[K', end='')
            if trial_num % 15 == 0: print(f'Running average after {trial_num} trials = {running_avg:.1f}')
    if dynamic_viz: plt.show()
    print(f'Completed {num_trials} in {(time()-start):.5}s')

def online_experiment(agent, num_trials, dynamic_viz=False):
    # TODO rewrite main 2048 function as to provide incremental rewards, states, and afterstates
    pass

# baseline(agent_class=DumbAgent(), num_trials=20000)

# evolutionary_experiment(
#     agent_class=EvolutionaryAgent(), \
#     generations=1000, \
#     num_agents=2, \
#     num_games=20, \
#     agents_that_survive=1, \
#     scale =0.05, \
#     dynamic_viz=True
# )

# offline_experiment(
#     agent=REINFORCEMonteCarloPolicyGradientAgent(), \
#     num_trials=10000, \
#     dynamic_viz=False
# )

