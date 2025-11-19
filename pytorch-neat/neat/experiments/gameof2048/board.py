import numpy as np
import random
from os import system
from copy import deepcopy
# from functions.repfuncs import log_modified

def log_modified(x, b=2):
    if x == 0: return 0
    return log(x, b)

def shiftLeft(board):
    # remove 0's in between numbers
    for i in range(4):
        nums, count = [], 0
        for j in range(4):
            if board[i][j] != 0:
                nums.append(board[i][j])
                count += 1
        board[i] = nums + [np.uint32(0) for _ in range(4-count)]

def shiftRight(board):
    # remove 0's in between numbers
    for i in range(4):
        nums, count = [], 0
        for j in range(4):
            if board[i][j] != 0:
                nums.append(board[i][j])
                count += 1
        board[i] = [np.uint32(0) for _ in range(4-count)] + nums

class Board:
    def __init__(self, given_board=None):
        if given_board is not None: self.board = given_board
        else: self.board = np.zeros((4,4), dtype=np.uint32)
        self.score = 0
        self.legal_moves = [i for i in range(4)]
        self.highest_tile = 0

    def __repr__(self): 
        system('clear')
        return_str = '|------------|\n'
        for i in range(4):
            row_str = '|'
            for j in range(4):
                row_str += ' ' + str(int(log_modified(self.board[i][j]))) + ' '
            return_str += row_str + '|\n'
        return_str += '|------------|\n' 
        return_str += f'     {self.score}  \n'
        return_str += '|------------|\n|      0     |\n|    3 . 1   |\n|      2     |\n|------------|'
        return return_str
        
    def start(self):
        self.spawn_tile()
        self.spawn_tile()

    def spawn_tile(self):
        # spawn tile ON self.board OBJ!!
        empty_cells = []
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    empty_cells.append((i,j))
        if len(empty_cells) > 0:
            cell_row, cell_col = empty_cells[np.random.randint(0,len(empty_cells))]
            if random.random() < 0.9:
                self.board[cell_row][cell_col] = np.uint32(2)
            else:
                self.board[cell_row][cell_col] = np.uint32(4)
        else:
            pass
        # generate legal moves, check for game over
        game_over = False
        self.legal_moves = []
        original_state = deepcopy(self.board)
        for direction in range(4):
            afterstate, reward = self.move_tiles(direction, apply=False)
            afterstate = np.array(afterstate)
            if not ((original_state == afterstate).all()):
                self.legal_moves.append(direction)
        if len(self.legal_moves) == 0: game_over = True
        return game_over
        
    def _moveLeft(self, board, apply=False):
        score = 0
        # initial shift
        shiftLeft(board)
        # merge cells
        for i in range(4):
            for j in range(3):
                if board[i][j] == board[i][j + 1] and board[i][j] != 0:
                    score += board[i][j]*2
                    board[i][j] *= 2
                    board[i][j + 1] = np.uint32(0)
                    j = 0
                    if board[i][j] > self.highest_tile and apply: self.highest_tile = board[i][j]
        # final shift
        shiftLeft(board)
        return board, score

    def _moveRight(self, board, apply=False):
        score = 0
        # initial shift
        shiftRight(board)
        # merge cells
        for i in range(4):
            for j in range(3, 0, -1):
                if board[i][j] == board[i][j - 1] and board[i][j] != 0:
                    score += board[i][j]*2
                    board[i][j] *= 2
                    board[i][j - 1] = np.uint32(0)
                    j = 0
                    if board[i][j] > self.highest_tile and apply: self.highest_tile = board[i][j]
        # final shift
        shiftRight(board)
        return board, score

    def _moveUp(self, board, apply=False):
        board = board.transpose()
        board, score = self._moveLeft(board, apply=apply)
        board = board.transpose()
        return board, score

    def _moveDown(self, board, apply=False):
        board = board.transpose()
        board, score = self._moveRight(board, apply=apply)
        board = board.transpose()
        return board, score

    def move_tiles(self, direction, apply=False):
        if apply: board = self.board # copy the reference to the argument board
        else: board = deepcopy(self.board) # make a copy of the argument board

        if direction == 0: # Up
            afterstate, score = self._moveUp(board, apply=apply)
        elif direction == 1: # Right
            afterstate, score = self._moveRight(board, apply=apply)
        elif direction == 2: # Down
            afterstate, score = self._moveDown(board, apply=apply)
        elif direction == 3: # Left
            afterstate, score = self._moveLeft(board, apply=apply)
        return deepcopy(afterstate), score

if __name__ == '__main__':


    board = np.array(
        [
            [  0,   2,   4,  16],
            [  0,   4,   8, 512],
            [  2,   8,  32, 256],
            [  8,  32, 128,  32]
        ]
    )
    board_obj = Board(board)

    print(board_obj.board)
    for action in range(4):
        print(f'action = {action}')
        print(board_obj.move_tiles(action))
        print('')
