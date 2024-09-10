import numpy as np
from functions.tuplefuncs import *

class nTupleNetwork:
    def __init__(self):
        self.tuple_map_class = TupleMap1()
        self.num_tuples = self.tuple_map_class.num_tuples
        self.num_tuple_reps = self.tuple_map_class.num_tuple_reps
        self.tuple_map = self.tuple_map_class.tuple_map
        self.lookup_array = [{} for _ in range(self.num_tuples)]

    def forward(self, activation):
        tuple_activations = self.tuple_map(activation) # TODO make sure we have exponents and not one hot
        afterstate_val = 0
        count = 0
        for tup_index, tup in enumerate(tuple_activations):
            for tup_rep in tup:
                try:
                    weight = self.lookup_array[tup_index][tup]
                except KeyError:
                    weight = 0
                    self.lookup_array[tup_index][tup] = weight  
                afterstate_val += weight
                count += 1
        # return (afterstate_val)
        return (afterstate_val/count)

    def backward(self, activation, label, learning_rate):
        delta_term = learning_rate*label
        # print(delta_term)
        tuple_activations = self.tuple_map(activation) # tupleify the afterstate
        # if abs(delta_term) > 100: print(delta_term)
        for tup_index, tup in enumerate(tuple_activations):
            temp_val = self.lookup_array[tup_index][tup]

            # self.lookup_array[tup_index][tup] += (delta_term/abs(delta_term))*min(abs(delta_term), 1000)
            self.lookup_array[tup_index][tup] += delta_term

