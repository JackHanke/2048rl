import numpy as np
import random
from os import system
from agents.humanagent import HumanAgent
from copy import deepcopy

def shiftLeft(board):
    # remove 0's in between numbers
    for i in range(4):
        nums, count = [], 0
        for j in range(4):
            if board[i][j] != 0:
                nums.append(board[i][j])
                count += 1
        board[i] = nums + [np.int32(0) for _ in range(4-count)]
        # board[i].extend([0] * (4 - count))

def shiftRight(board):
    # remove 0's in between numbers
    for i in range(4):
        nums, count = [], 0
        for j in range(4):
            if board[i][j] != 0:
                nums.append(board[i][j])
                count += 1
        board[i] = [np.int32(0) for _ in range(4-count)] + nums

class Board:
    def __init__(self):
        self.board = np.zeros((4,4), dtype=np.int32)
        self.score = 0
        self.legal_moves = [i for i in range(4)]
        # TODO largest tile tracking

    def __repr__(self): 
        system('clear')
        return_str = '|------------|\n'
        for i in range(4):
            row_str = '|'
            for j in range(4):
                row_str += ' ' + str(int(self.board[i][j])) + ' '
            return_str += row_str + '|\n'
        return_str += '|------------|\n' 
        return_str += f'| Score = {self.score}  |\n'
        return_str += '|------------|\n|      0     |\n|    3 . 1   |\n|      2     |\n|------------|'
        return return_str
        
    def start(self):
        self.spawn_tile()
        self.spawn_tile()

    def spawn_tile(self):
        # spawn tile 
        empty_cells = []
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    empty_cells.append((i,j))
        if len(empty_cells) > 0:
            cell_row, cell_col = empty_cells[np.random.randint(0,len(empty_cells))]
            if random.random() < 0.9:
                self.board[cell_row][cell_col] = np.int32(2)
            else:
                self.board[cell_row][cell_col] = np.int32(4)
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
        if not apply: board = deepcopy(board)
        # initial shift
        shiftLeft(board)
        # merge cells
        for i in range(4):
            for j in range(3):
                if board[i][j] == board[i][j + 1] and board[i][j] != 0:
                    score += board[i][j]*2
                    board[i][j] *= 2
                    board[i][j + 1] = np.int32(0)
                    j = 0
        # final shift
        shiftLeft(board)
        if apply: self.board = board
        return board, score

    def _moveRight(self, board, apply=False):
        score = 0
        if not apply: board = deepcopy(board)
        # initial shift
        shiftRight(board)
        # merge cells
        for i in range(4):
            for j in range(3, 0, -1):
                if board[i][j] == board[i][j - 1] and board[i][j] != 0:
                    score += board[i][j]*2
                    board[i][j] *= 2
                    board[i][j - 1] = np.int32(0)
                    j = 0
        # final shift
        shiftRight(board)
        if apply: self.board = board
        return board, score

    def _moveUp(self, board, apply=False):
        if not apply: board = deepcopy(board)
        board = board.transpose()
        board, score = self._moveLeft(board)
        board = board.transpose()
        if apply: self.board = board
        return board, score

    def _moveDown(self, board, apply=False):
        if not apply: board = deepcopy(board)
        board = board.transpose()
        board, score = self._moveRight(board)
        board = board.transpose()
        if apply: self.board = board
        return board, score

    def move_tiles(self, direction, apply=False):
        if direction == 0: # Up
            afterstate, score = self._moveUp(self.board, apply=apply)
        elif direction == 1: # Right
            afterstate, score = self._moveRight(self.board, apply=apply)
        elif direction == 2: # Down
            afterstate, score = self._moveDown(self.board, apply=apply)
        elif direction == 3: # Left
            afterstate, score = self._moveLeft(self.board, apply=apply)
        return afterstate, int(score)

class Gameof2048:
    def __init__(self, agent):
        self.game_over = False
        self.board = Board()
        self.agent = agent

    def play(self, verbose=False):
        afterstate = self.board.board
        self.board.start()
        if self.agent.type == 'human': print(self.board)
        while not self.game_over:
            if self.agent.type == 'human':
                # print(f'legal moves = {self.board.legal_moves}')
                direction = int(input()) # TODO type error handling
                while (direction not in self.board.legal_moves) or (direction not in (0,1,2,3)):
                    direction = int(input('Enter 0 (Up) 1 (Right) 2 (Down) 3 (Left)\n'))
            else:

                afterstates = []
                for action_emb in self.board.legal_moves:
                    temp_afterstate, reward = self.board.move_tiles(action_emb, apply=False)
                    afterstates.append((action_emb, reward, temp_afterstate))
                # TODO remove
                # print('state:')
                # print(self.board.board)
                direction = self.agent.choose(
                    state=self.board.board,
                    afterstates=afterstates
                )
                # print(f'agent chose {direction}')
                # input('>>>')
                self.agent.update(
                    afterstate=afterstate,
                    state=self.board.board,
                    chosen_action=direction,
                    afterstates=afterstates
                )

            afterstate, reward = self.board.move_tiles(direction, apply=True)
            self.game_over = self.board.spawn_tile()
            self.board.score += reward
            if self.agent.type == 'human': print(self.board)
        if verbose: print(f'Final Score = {self.board.score}')
        return self.board.score    

if __name__ == '__main__':
    # TODO make sure only legal moves can be made
    game = Gameof2048(agent=HumanAgent())
    game.start()
