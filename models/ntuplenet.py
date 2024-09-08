import numpy as np

def tuple_map(board):
    # board = [original_board[(4*k):(4*(k+1))] for k in range(4)]
    # print(board)
    tuple_list = []
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

def new_tuple_map(board):
    tuple_list = []
    tuple_list.append((board[0][0], board[0][1], board[0][2], board[0][3],board[1][0], board[1][1], board[1][2], board[1][3]))
    tuple_list.append((board[1][0], board[1][1], board[1][2], board[1][3],board[2][0], board[2][1], board[2][2], board[2][3]))
    tuple_list.append((board[2][0], board[2][1], board[2][2], board[2][3],board[3][0], board[3][1], board[3][2], board[3][3]))
    return tuple_list

class nTupleNetwork:
    def __init__(self):
        self.num_tuples = 3
        self.tuple_map = new_tuple_map
        self.lookup_array = [{} for _ in range(self.num_tuples)]

    def forward(self, activation):
        tuple_activations = self.tuple_map(activation) # TODO make sure we have exponents and not one hot
        afterstate_val = 0
        for tup_index, tup in enumerate(tuple_activations):
            try:
                weight = self.lookup_array[tup_index][tup]
            except KeyError:
                weight = 0
                self.lookup_array[tup_index][tup] = weight  
            afterstate_val += weight
        #     print(f'tup {tup_index} = {tup} has weight {weight}')
        # print('>>')
        # if afterstate_val > 1000:
        #     for tup_index, tup in enumerate(tuple_activations):
        #         print(f'val for {tup} = {self.lookup_array[tup_index][tup]}')
        #     print(activation)
        #     input()

        # if afterstate_val/len(self.lookup_array) > 50000: 
        #     print(activation)
        #     input('your code sucks dude')
        return afterstate_val/len(self.lookup_array)

    def backward(self, activation, label, learning_rate):
        delta_term = learning_rate*label
        # print(delta_term)
        tuple_activations = self.tuple_map(activation) # tupleify the afterstate
        # if abs(delta_term) > 100: print(delta_term)
        for tup_index, tup in enumerate(tuple_activations):
            temp_val = self.lookup_array[tup_index][tup]
            if delta_term == 0:
                pass
            elif temp_val > 50000:
                # print('happened idiot')
                pass
            # elif delta_term != 0 and temp_val < 50000:
            elif delta_term != 0:
                # self.lookup_array[tup_index][tup] += (delta_term/abs(delta_term))*min(abs(delta_term), 1000)
                self.lookup_array[tup_index][tup] += delta_term
                if np.isnan(self.lookup_array[tup_index][tup]):
                    print(f' temp val = {temp_val}')
                    print(tup)
                    print(f' delta term = {delta_term}')
                    input('Press ENTER to continue with the bad code')



