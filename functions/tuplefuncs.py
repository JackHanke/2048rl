'''
    00 01 02 03
    10 11 12 13
    20 21 22 23
    30 31 32 33
'''
# TODO it's possible the ordering of the tuples matters
class TupleMap1:
    def __init__(self):
        self.num_tuples = 4
        self.num_tuple_reps = 1
    def tuple_map(self, board):
        tuple_list = [
            # corner 2x2s
            (
                (board[0][0], board[0][1], board[1][0], board[1][1]),
                (board[2][0], board[2][1], board[3][0], board[3][1]),
                (board[0][2], board[0][3], board[1][2], board[1][3]),
                (board[2][2], board[2][3], board[3][2], board[3][3]),
                (board[1][1], board[1][2], board[2][1], board[2][2])
            ),
            # edge 2x2s
            (
                (board[0][1], board[0][2], board[1][1], board[1][2]),
                (board[1][0], board[1][1], board[2][0], board[2][1]),
                (board[1][2], board[1][3], board[2][2], board[2][3]),
                (board[2][1], board[2][2], board[3][1], board[3][2])
            ),
            # horizontal bars
            (
                (board[0][0], board[1][0], board[2][0], board[3][0]),
                (board[0][1], board[1][1], board[2][1], board[3][1]),
                (board[0][2], board[1][2], board[2][2], board[3][2]),
                (board[0][3], board[1][3], board[2][3], board[3][3])
            ),
            # vertical bars
            (
                (board[0][0], board[0][1], board[0][2], board[0][3]),
                (board[1][0], board[1][1], board[1][2], board[1][3]),
                (board[2][0], board[2][1], board[2][2], board[2][3]),
                (board[3][0], board[3][1], board[3][2], board[3][3])
            )
        ]
        return tuple_list

class TupleMap2:
    def __init__(self):
        self.num_tuples = 3
        self.num_tuple_reps = 1
    def tuple_map(self, board):
        tuple_list = [
            (
                (board[0][0], board[0][1], board[0][2], board[0][3],board[1][0], board[1][1], board[1][2], board[1][3])
            ),
            (
                (board[1][0], board[1][1], board[1][2], board[1][3],board[2][0], board[2][1], board[2][2], board[2][3])
            ),
            (
                (board[2][0], board[2][1], board[2][2], board[2][3],board[3][0], board[3][1], board[3][2], board[3][3])
            )
            
        ]
        return tuple_list

class TupleMap3:
    def __init__(self):
        self.num_tuples = 2
        self.num_tuple_reps = 12
    def tuple_map(self, board):
        tuple_list = [
            (
                (board[0][0], board[0][1], board[0][2], board[1][0], board[1][1], board[1][2]),
                (board[0][1], board[0][2], board[0][3], board[1][1], board[1][2], board[1][3]),
                (board[2][0], board[2][1], board[2][2], board[3][0], board[3][1], board[3][2]),
                (board[2][1], board[2][2], board[2][3], board[3][1], board[3][2], board[3][3]),
                (board[0][0], board[0][1], board[1][0], board[1][1], board[2][0], board[2][1]),
                (board[0][2], board[0][3], board[1][2], board[1][3], board[2][2], board[2][3]),
                (board[1][0], board[1][1], board[2][0], board[2][1], board[3][0], board[3][1]),
                (board[1][2], board[1][3], board[2][2], board[2][3], board[3][2], board[3][3])
            ),
            (
                (board[1][0], board[1][1], board[1][2], board[2][0], board[2][1], board[2][2]),
                (board[1][1], board[1][2], board[1][3], board[2][1], board[2][2], board[2][3]),
                (board[0][1], board[0][2], board[1][1], board[1][2], board[2][1], board[2][2]),
                (board[1][1], board[1][2], board[2][1], board[2][2], board[3][1], board[3][2])
            )
        ]
        return tuple_list
