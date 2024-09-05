import numpy as np
import random
from os import system
from game.logic import *
from agents.humanagent import HumanAgent
from copy import deepcopy

class Board:
    def __init__(self):
        self.board = np.zeros((4,4))
        self.score = 0

    def __repr__(self): # TODO this needs to be better
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
        empty_cells = []
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    empty_cells.append((i,j))
        cell_row, cell_col = empty_cells[np.random.randint(0,len(empty_cells))]
        if random.random() < 0.9:
            self.board[cell_row][cell_col] = 2
        else:
            self.board[cell_row][cell_col] = 4

    def _moveLeft(self, board, apply=False):
        score = 0
        if apply: board = deepcopy(self.board)
        # initial shift
        shiftLeft(board)
        # merge cells
        for i in range(4):
            for j in range(3):
                if board[i][j] == board[i][j + 1] and board[i][j] != 0:
                    score += board[i][j]*2
                    board[i][j] *= 2
                    board[i][j + 1] = 0
                    j = 0
        # final shift
        shiftLeft(board)
        if apply: self.board = board
        return board, score

    def _moveRight(self, board, apply=False):
        score = 0
        if apply: board = deepcopy(self.board)
        # initial shift
        shiftRight(board)
        # merge cells
        for i in range(4):
            for j in range(3, 0, -1):
                if board[i][j] == board[i][j - 1] and board[i][j] != 0:
                    score += board[i][j]*2
                    board[i][j] *= 2
                    board[i][j - 1] = 0
                    j = 0
        # final shift
        shiftRight(board)
        if apply: self.board = board
        return board, score

    def _moveUp(self, board, apply=False):
        if apply: board = deepcopy(self.board)
        board = rotateLeft(board)
        board, score = self._moveLeft(board)
        board = rotateRight(board)
        if apply: self.board = board
        return board, score

    def _moveDown(self, board, apply=False):
        if apply: board = deepcopy(self.board)
        board = rotateLeft(board)
        board, score = self._moveLeft(board)
        shiftRight(board)
        board = rotateRight(board)
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

def game(agent):
    if agent.type == 'human': system('clear')
    game_over = False
    board = Board()
    board.start()
    if agent.type == 'human': print(board)
    while not game_over:
        if agent.type == 'human':
            direction = input()
            while direction not in ('0','1','2','3'):
                direction = input('Enter 0 (Up) 1 (Right) 2 (Down) 3 (Left)')
            direction = int(direction)
        else:
            pass

        if agent.type == 'human': system('clear')
        afterstate, reward = board.move_tiles(direction, apply=True)
        board.spawn_tile()
        board.score += reward
        if agent.type == 'human': print(board)

    if verbose: print(f'Final Score = {board.score}')
            

game(HumanAgent())