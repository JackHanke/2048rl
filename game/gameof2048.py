from time import sleep
from copy import deepcopy

from game.board import Board

class Gameof2048:
    def __init__(self, agent, watch:bool = False):
        self.game_over = False
        self.board = Board()
        self.agent = agent
        self.watch = watch
        self.moves = 0
        # 
        self.do_display = self.agent.type == 'human' or self.watch
        self.needs_afterstates = False

        """
        intial_board: [board array] // initial condition for the board
        gameplay: [
            {
                move_made: int 0 1 2 3 // move made 
                new_tile_idx: tup(int 0 1 2 3, int 0 1 2 3) // place where the new tile spawned
                new_tile_val: int 2 4 // value of new tile spawned
                move_idx: int // index of move in game 
                reward: int // score 
            }
        ]
        """
        self.board.start()
        self.gameplay = {
            'initialboard': deepcopy(self.board.board.tolist()),
            'gameplay': []
        }

    def play(self, verbose:bool = False):
        afterstate = deepcopy(self.board.board)
        if self.do_display: print(self.board)
        
        while not self.game_over:
            if self.agent.type == 'human':
                direction = int(input()) # TODO type error handling
                while (direction not in self.board.legal_moves) or (direction not in (0,1,2,3)):
                    direction = int(input('Enter 0 (Up) 1 (Right) 2 (Down) 3 (Left)\n'))
            else:
                if self.needs_afterstates:
                    afterstates = []
                    for move in self.board.legal_moves:
                        temp_afterstate, reward = self.board.move_tiles(move, apply=False)
                        afterstates.append((move, reward, temp_afterstate))
                    direction = self.agent.choose(
                        state = self.board.board,
                        afterstates = afterstates
                    )
                else:
                    direction = self.agent.choose(
                        state = self.board.board
                    )

            afterstate, reward = self.board.move_tiles(direction, apply=True)
            
            self.game_over, new_tile_coords, new_tile_value = self.board.spawn_tile()
            
            self.moves += 1

            self.gameplay['gameplay'].append({
                'move_made': direction,
                'new_tile_idx': new_tile_coords,
                'new_tile_val': int(new_tile_value),
                'move_idx': self.moves,
                'reward': int(reward),
            })

            self.board.score += reward
            if self.do_display: print(self.board)
            if self.watch: 
                print(f'Agent chose: {direction}')
                sleep(0.1)
        
        if verbose: print(f'Final Score = {self.board.score}')
        return self.board.score    
