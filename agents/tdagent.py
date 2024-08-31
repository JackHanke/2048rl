# from models import ArtificialNeuralNetwork
from models.linear import Linear
from functions.environmentfuncs import one_hot_state_rep
from functions.rlfuncs import epsilon_greedy
# read this: https://web.stanford.edu/group/pdplab/pdphandbook/handbookch10.html

# 
class MonteCarloApproxAgent:
    def __init__(self):
        self.name = 'Monte Carlo Approx Agent'
        self.type = 'offline'
        self.lmbda = 0
        self.n_step = 1
        self.discounting_param = 1
        self.epsilon = 0.1
        self.learning_rate = -0.01
        self.state_rep_function = one_hot_state_rep
        self.state_value_function_approx = Linear(dims=(256,1), seed=1)

    def update(self):
        return_val = 0
        for t in range(len(self.state_history)):
            current_state = self.state_history[t]
            current_state = np.array([current_state]).transpose()
            current_action = self.action_history[t]
            predicted_state_val = self.state_value_function_approx._forward(current_state)
            return_val = sum([(self.gamma**(k-t-1)) * (self.reward_history[k]) for k in range(t+1,len(self.state_history))])
            # predicted_state_val = self.baseline_function._forward(current_state)
            delta = (return_val - predicted_state_val)
            # learning_rate_term = self.learning_rate*return_val*((self.gamma)**t)
            lr_term = self.policy_learning_rate*delta
            self.policy_function._backward(
                state=current_state, \
                label=None, \
                learning_rate=lr_term
            )
        # flush history
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]

    def choose(self, state):
        # generate evaluation of 
        predicted_state_vals = []
        return epsilon_greedy(predicted_state_vals, self.epsilon)
