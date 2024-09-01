import numpy as np
import random

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
        return_str += '|------------|' 
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

    def move_tiles(self, direction):
        pass

class Player:
    def __init__(self):
        self.type = 'human'

def game(agent):
    game_over = False
    board = Board()
    board.start()
    print(board)
    while not game_over:
        
        
        if agent.type is 'human':
            direction = input()
            print(direction)
        else:
            pass

        reward = board.move_tiles(direction)
        board.spawn_tile()
        print(board)
            

game(Player())