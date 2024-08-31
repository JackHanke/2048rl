

class LinearSoftmax:
    def __init__(self):
        np.random.seed(1)
        self.weights = np.random.normal(loc=0, scale=1/16, size=(4 ,(16*16)))

    def _forward(self, activation):
        return softmax(np.dot(self.weights, activation))

    def _backward(self, state, label, learning_rate):
        activation = self._forward(state)
        gradient = np.zeros((4,(16*16)))
        gradient[label] = state.transpose()
        gradient -= np.dot(activation, state.transpose())
        
        self.weights -= learning_rate*gradient
        return 0, 0

