import random

# takes array arr 
# returns uniform random sample of index from arrray of values that are tied for largest value in arr
# if return_arr == True then return the list of possible values instead of sample from one
def better_argmax(arr, return_arr=False):
    maxval = -float('inf')
    best_indices_arr =[]
    for index, val in enumerate(arr):
        if val > maxval:
            maxval = val
            best_indices_arr = [index]
        elif val == maxval:
            best_indices_arr.append(index)
    max_index = np.random.randint(low=0,high=len(best_indices_arr))
    if return_arr: return best_indices_arr
    return best_indices_arr[max_index]

# returns better_argmax for arr with probability 1-epsilon, otherwise returns random index
def epsilon_greedy(arr, epsilon):
    greedy_action = better_argmax(arr)
    if np.random.uniform(0,1) <  epsilon:
        random_action = greedy_action
        while random_action == greedy_action:
            random_action = np.random.randint(low=0,high=((self.k)))
        return random_action
    else:
        return greedy_action

# takes array arr that is a discrete probability dis [p_0, p_1 ... p_n]
# returns the index i with probability p_i
def prob_argmax(arr): 
    return int(np.random.choice([i for i in range(len(vec))],1,p=arr)[0])
        