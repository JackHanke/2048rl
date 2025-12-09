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
