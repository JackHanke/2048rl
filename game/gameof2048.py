from time import sleep
from game.board import Board
from copy import deepcopy

class Gameof2048:
    def __init__(self, agent, watch=False):
        self.game_over = False
        self.board = Board()
        self.agent = agent
        self.watch=watch
        self.gameplay = [deepcopy(self.board.board.tolist())]

    def play(self, verbose=False):
        afterstate = deepcopy(self.board.board)
        self.board.start()
        if self.agent.type == 'human' or self.watch: print(self.board)
        while not self.game_over:
            self.gameplay.append(deepcopy(self.board.board.tolist()))
            if self.agent.type == 'human':
                direction = int(input()) # TODO type error handling
                while (direction not in self.board.legal_moves) or (direction not in (0,1,2,3)):
                    direction = int(input('Enter 0 (Up) 1 (Right) 2 (Down) 3 (Left)\n'))
            else:
                afterstates = []
                for action_emb in self.board.legal_moves:
                    temp_afterstate, reward = self.board.move_tiles(action_emb, apply=False)
                    afterstates.append((action_emb, reward, temp_afterstate))
                direction = self.agent.choose(
                    state=self.board.board,
                    afterstates=afterstates
                )
                if 'online' in self.agent.type:
                    # print('afterstate:')
                    # print(afterstate)
                    # print('state:')
                    # print(self.board.board)
                    # input()
                    self.agent.update(
                        afterstate=afterstate,
                        state=self.board.board,
                        chosen_action=direction,
                        afterstates=afterstates
                    )
            afterstate, reward = self.board.move_tiles(direction, apply=True)
            
            self.gameplay += [int(direction), int(reward), deepcopy(afterstate.tolist())]
            self.game_over = self.board.spawn_tile()
            self.board.score += reward
            if self.agent.type == 'human' or self.watch: print(self.board)
            if self.watch: 
                print(f'Agent chose: {direction}')
                sleep(0.1)
        
        if 'offline' in self.agent.type:
            self.agent.update(
                afterstate=afterstate,
                state=self.board.board,
                chosen_action=direction,
                afterstates=afterstates
            )
        if verbose: print(f'Final Score = {self.board.score}')
        return self.board.score    
