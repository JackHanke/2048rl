import numpy as np
from functions.repfuncs import log_modified
import json

def tuple_codify(tup):
    return_str = ''
    for val in tup:
        exponent_str = str(int(log_modified(val)))
        if len(exponent_str) == 1:
            return_str += '0' + exponent_str
        elif len(exponent_str) == 2:
            return_str += exponent_str
    return return_str

class nTupleNetwork:
    def __init__(self, tuple_map_class, load_loc=None):
        self.tuple_map_class = tuple_map_class
        self.num_tuples = self.tuple_map_class.num_tuples
        self.tuple_map = self.tuple_map_class.tuple_map
        if load_loc is None: self.lookup_array = [{} for _ in range(self.num_tuples)]
        else: 
            with open(load_loc, 'r') as fin:
                self.lookup_array = json.load(fin)

    def forward(self, activation):
        tuple_activations = self.tuple_map(activation) # TODO make sure we have exponents and not one hot
        afterstate_val = 0
        for tup_index, tup_collection in enumerate(tuple_activations):
            for tup_rep in tup_collection:
                tup_code = tuple_codify(tup_rep)
                try:
                    weight = self.lookup_array[tup_index][tup_code]
                except KeyError:
                    weight = 0
                    self.lookup_array[tup_index][tup_code] = weight  
                afterstate_val += weight
        return afterstate_val

    def backward(self, activation, label, learning_rate):
        delta_term = learning_rate*label
        tuple_activations = self.tuple_map(activation) # tupleify the afterstate
        for tup_index, tup_collection in enumerate(tuple_activations):
            for tup_rep in tup_collection:
                tup_code = tuple_codify(tup_rep)
                self.lookup_array[tup_index][tup_code] += delta_term

    def get_num_params(self):
        num_params = 0
        for tup_index, tup in enumerate(tuple_activations):
            num_params += len(self.lookup_array[tup_index].keys())
        return num_params