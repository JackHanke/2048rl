
def tuple_map(original_board):
    board = [original_board[(4*k):(4*(k+1))] for k in range(4)]
    tuple_list = []
    print(board)
    # corner 2x2s
    tuple_list.append((board[0][0], board[0][1], board[1][0], board[1][1]))
    tuple_list.append((board[2][0], board[2][1], board[3][0], board[3][1]))
    tuple_list.append((board[0][2], board[0][3], board[1][2], board[1][3]))
    tuple_list.append((board[2][2], board[2][3], board[3][2], board[3][3]))
    tuple_list.append((board[1][1], board[1][2], board[2][1], board[2][2]))
    # edge 2x2s
    tuple_list.append((board[0][1], board[0][2], board[1][1], board[1][2]))
    tuple_list.append((board[1][0], board[1][1], board[2][0], board[2][1]))
    tuple_list.append((board[1][2], board[1][3], board[2][2], board[2][3]))
    tuple_list.append((board[2][1], board[2][2], board[3][1], board[3][2]))
    # horizontal bars
    tuple_list.append((board[0][0], board[1][0], board[2][0], board[3][0]))
    tuple_list.append((board[0][1], board[1][1], board[2][1], board[3][1]))
    tuple_list.append((board[0][2], board[1][2], board[2][2], board[3][2]))
    tuple_list.append((board[0][3], board[1][3], board[2][3], board[3][3]))
    # vertical bars
    tuple_list.append((board[0][0], board[0][1], board[0][2], board[0][3]))
    tuple_list.append((board[1][0], board[1][1], board[1][2], board[1][3]))
    tuple_list.append((board[2][0], board[2][1], board[2][2], board[2][3]))
    tuple_list.append((board[3][0], board[3][1], board[3][2], board[3][3]))

    return tuple_list

class nTupleNetwork:
    def __init__(self, num_tuples):
        self.lookup_array = [{} for _ in range(num_tuples)]

    def forward(self, activation):
        tuple_activations = tuple_map(activation) # TODO make sure we have exponents and not one hot
        afterstate_val = 0
        for tup_index, tup in enumerate(tuple_activations):
            try:
                weight = self.lookup_array[tup_index][tup]
            except KeyError:
                weight = 0
                self.lookup_array[tup_index][tup] = weight  
            afterstate_val += weight
        return afterstate_val

    def backward(self, activation, label, learning_rate):
        delta_term = learning_rate*label
        tuple_activations = tuple_map(activation) # tupleify the afterstate
        for tup_index, tup in enumerate(tuple_activations):
            self.lookup_array[tup_index][tup] += delta_term


