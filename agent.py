from macht.macht.term import main
from statistics import stdev, median, mean
from time import time
import random
import matplotlib.pyplot as plt
import numpy as np
from model import Network, sigmoid, sigmoid_prime, softmax, LinearSoftmax, relu, relu_prime, leaky_relu, leaky_relu_prime

def simple_exponent_state_rep(grid):
    rep = [0 for _ in range(16)]
    for row_index, row in enumerate(grid):
        for column_index, tile in enumerate(row):
            if tile is None:
                rep[(row_index * 4)  + column_index] = 0
            else:
                rep[(row_index * 4)  + column_index] = tile.exponent/16
    return rep

def one_hot_state_rep(grid):
    # exponent 0 1 2 3 4 5 6 7 8 9
    # encoding 0 0 0 0 0 0 0 0 1 0 
    rep = []
    for row_index, row in enumerate(grid):
        for column_index, tile in enumerate(row):
            if tile is None:
                rep += [1] + [0 for _ in range(15)] 
            else:
                temp_vec = [0 for _ in range(16)]
                temp_vec[tile.exponent] = 1
                rep += temp_vec
    return rep

# the agent that learns nothing an acts randomly
class DumbAgent:
    def __init__(self):
        self.name = 'Dumb Agent'
        self.type = 'offline'
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]
        self.state_representation_function = simple_exponent_state_rep
    def choose(self, state, invalid_moves): return random.choice((0, 1, 2, 3))
    def update(self): pass

class EvolutionaryAgent:
    def __init__(self):
        self.name = 'Evolutionary Agent(s)'
        self.type = 'evolutionary'
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]
        self.state_representation_function = one_hot_state_rep
        self.policy_function = Network(
            dims=(256,30,30,30,4), \
            activation_funcs = [
                (sigmoid, sigmoid_prime), \
                (sigmoid, sigmoid_prime), \
                (sigmoid, sigmoid_prime), \
                (softmax, softmax) \
            ]
        ) # structure of function mappping states to actions for all agents

    def choose(self, state, invalid_moves):
        def prob_argmax(vec): return int(np.random.choice([i for i in range(len(vec))],1,p=vec)[0])
        state = np.array([state]).transpose()
        activation = self.policy_function._forward(state).transpose()[0]
        # invalid action masking
        for action, prob in enumerate(activation):
            if action in invalid_moves:
                activation[action] = 0
        normalize = sum(activation)
        for action, prob in enumerate(activation):
            activation[action] = prob/normalize
        return_val = prob_argmax(activation)
        return return_val
    
    def mutate(self, scale):
        for weights_index, weights in enumerate(self.policy_function.weights):
            if weights_index > 1: self.policy_function.weights[weights_index] += np.random.normal(loc=0, scale=scale, size=weights.shape)

        for biases_index, biases in enumerate(self.policy_function.biases):
            if biases_index > 1: self.policy_function.biases[biases_index] += np.random.normal(loc=0, scale=scale, size=biases.shape)

class REINFORCEMonteCarloPolicyGradientAgent:
    def __init__(self):
        self.name = 'REINFORCE Monte Carlo Policy Gradient Agent'
        self.type = 'offline'
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]
        self.state_representation_function = one_hot_state_rep
        self.gamma = 0.99 # discounting factor for reward
        self.policy_function = LinearSoftmax()
        self.policy_learning_rate = -0.00001 # negative to perform stochastic gradient ASCENT
        # self.policy_function = Network(
        #     dims=(256,80,4), \
        #     activation_funcs = [
        #         (leaky_relu, leaky_relu_prime), \
        #         (softmax, softmax) \
        #     ], \
        #     seed=1
        # )
        self.baseline_learning_rate = -0.001 # negative to perform stochastic gradient ASCENT
        self.baseline_function = Network(
            dims=(256,80,1), \
            activation_funcs = [
                (leaky_relu, leaky_relu_prime), \
                (leaky_relu, leaky_relu_prime)
            ], \
            seed=1
        )

    def update(self):
        return_val = 0

        for t in range(len(self.state_history)):
            current_state = self.state_history[t]
            current_state = np.array([current_state]).transpose()
            current_action = self.action_history[t]
            
            policy_eval = self.policy_function._forward(current_state)
            
            return_val = sum([(self.gamma**(k-t-1)) * (self.reward_history[k]) for k in range(t+1,len(self.state_history))])
            # predicted_state_val = self.baseline_function._forward(current_state)
            predicted_state_val = 0
            # print(f'return_val at time {t} with gamma={self.gamma} is {return_val} ')
            # print(f'predicted_state_val at time {t} = {predicted_state_val}')
            delta = (return_val - predicted_state_val)/1
            # delta = (return_val - 30)
            # learning_rate_term = self.learning_rate*return_val*((self.gamma)**t)
            # dude = max(policy_eval[current_action][0],0.1)
            dude = policy_eval[current_action][0]
            # policy_lr_term = self.policy_learning_rate*delta*((self.gamma)**t)/dude
            policy_lr_term = self.policy_learning_rate*delta*((self.gamma)**t)
            baseline_lr_term = self.policy_learning_rate*delta
            # learning_rate_term = self.learning_rate*return_val*((self.gamma)**t)
            # print(learning_rate_term)
            # input()

            self.policy_function._backward(
                current_state, \
                current_action, \
                policy_lr_term
            )

            # self.baseline_function._backward(
            #     current_state, \
            #     current_action, \
            #     baseline_lr_term
            # )

        # flush history
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]


    def choose(self, state, invalid_moves):
        def prob_argmax(vec): return int(np.random.choice([i for i in range(len(vec))],1,p=vec)[0])
        state = np.array([state]).transpose()
        activation = self.policy_function._forward(state).transpose()[0]
        # invalid action masking
        for action, prob in enumerate(activation):
            if action in invalid_moves:
                activation[action] = 0
        normalize = sum(activation)
        # if normalize < 0.000000000001:
            # return np.argmax(activation)
        # print(activation)
        # input()
        for action, prob in enumerate(activation):
            activation[action] = prob/normalize
        return_val = prob_argmax(activation)
        return return_val

class ActorCriticAgent:
    def __init__(self):
        self.name = 'Actor Critic Agent'

        self.value_function = 0
        self.policy_function = 0
        self.performance_function = 0

    def update(self):
        pass

    def choose(self):
        pass

# mean ~ 1096
# stdev ~ 549
# median ~ 1048
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

        # weights.append(max_weight)
        # running_avg_weights = mean(weights)

        # grads.append(max_grad)
        # running_avg_grads = mean(grads)
        if dynamic_viz:

            # plt.subplot(1,2,1)
            plt.scatter(trial_num, final_score, c='red')
            plt.scatter(trial_num, running_avg, c='orange')
            plt.title(f'2048 {agent.name} Score')
            plt.xlabel(f'Trial Number (Completed in {(time()-start):.2f}s)')
            plt.ylabel(f'Final Score, Running Average = {running_avg:.1f} Points')
            

            # plt.subplot(1,2,2)
            # plt.scatter(trial_num, best_tile, c='pink')
            # plt.title(f'2048 {agent.name} best tile achieved')
            # plt.xlabel(f'Trial Number (Completed in {(time()-start):.2f}s)')
            # plt.ylabel(f'Tile Exponent')

            # plt.subplot(1,2,2)
            # plt.scatter(trial_num, max_weight, c='yellow')
            # plt.scatter(trial_num, running_avg_weights, c='green')
            # plt.title(f'Max Weights for {agent.name}')
            # plt.xlabel(f'Trial Number (Completed in {(time()-start):.2f}s)')
            # plt.ylabel(f'Max Weight, Running Average = {running_avg:.1f} Points')

            # plt.subplot(3,1,3)
            # plt.scatter(trial_num, max_grad, c='blue')
            # plt.scatter(trial_num, running_avg_grads, c='purple')
            # plt.title(f'Max Grad for {agent.name}')
            # plt.xlabel(f'Trial Number (Completed in {(time()-start):.2f}s)')
            # plt.ylabel(f'Max Grad Norm, Running Average = {running_avg:.1f} Points')

            plt.pause(0.00001)
    if dynamic_viz: plt.show()
    print(f'Completed {num_trials} in {(time()-start):.5}s')

def online_experiment(agent, num_trials, dynamic_viz=False):
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

offline_experiment(
    agent=REINFORCEMonteCarloPolicyGradientAgent(), \
    num_trials=10000, \
    dynamic_viz=True
)


