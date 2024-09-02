import sys
import random
import signal
import argparse
from functools import partial, reduce
from itertools import chain
from time import time
import blessed

from .. import save
from ..grid import Direction, Actions
from .grid import Grid
from .tile import Tile

up, left = ('w', 'k', 'KEY_UP', 0), ('a', 'h', 'KEY_LEFT', 1)
down, right = ('s', 'j', 'KEY_DOWN', 2), ('d', 'l', 'KEY_RIGHT', 3)

grid_moves = {}
for keys, direction in zip((up, left, down, right), Direction):
    grid_moves.update(dict.fromkeys(keys, direction))


def grid_dimension(string):
    rows, _, cols = string.partition('x')
    try:
        return {'rows': int(rows), 'cols': int(cols)}
    except ValueError:
        raise argparse.ArgumentTypeError(
            "grid dimension should look like: '4x4'")


parser = argparse.ArgumentParser(
    description="A game with the objective of merging tiles by moving them.",
    epilog="Use the arrow, wasd or hjkl keys to move the tiles.")
parser.add_argument('grid_dims', metavar='GRID_DIMENSIONS',
                    type=grid_dimension, nargs='*',
                    help="Dimensions used fnor grid(s), default: '4x4'")
parser.add_argument('-b', '--base', metavar='N', type=int,
                    help="base value of all tiles")
parser.add_argument('-r', '--resume', metavar='SAVE_FILE', nargs='?',
                    default=False, const=None,
                    help="resume previous game. SAVE_FILE is used to save to "
                    "and resume from. Specifying grid dimensions and/or base "
                    "starts a new game without resuming from SAVE_FILE.")
parser.add_argument('-a', '--auto', help="Run game on auto.", action='store_true')
parser.add_argument('-v', '--viz', help="Get visual output.", action='store_true')

def draw_score(score, term, end=False):
    msg = "score: " + str(score)
    with term.location(term.width // 2 - len(msg) // 2, 0):
        print(term.bold_on_red(msg) if end else term.bold(msg))


def term_resize(term, grids):
    print(term.clear())

    max_width = (term.width - (len(grids) + 1) * 2) // len(grids)

    for grid in grids:
        for tile_height in range(10, 2, -1):
            grid.tile_height, grid.tile_width = tile_height, tile_height * 2

            if grid.height + 1 < term.height and grid.width <= max_width:
                break
        else:
            with term.location(0, 0):
                print(term.red("terminal size is too small;\n"
                               "please resize the terminal"))
            return False  # game can not continue until after another resize

    margin = (term.width - sum(g.width for g in grids) -
              (len(grids) - 1) * 2) // 2
    for grid_idx, grid in enumerate(grids):
        grid.x = margin + sum(g.width for g in grids[:grid_idx]) + grid_idx * 2

        grid.draw()
        grid.update_tiles()
        grid.draw_tiles()

    return True

def main(args=None, agent=None):
    global do_resize
    do_resize = True

    term = blessed.Terminal()
    term_too_small = False
    game_over = False

    def on_resize(signal, frame):
        global do_resize
        do_resize = True
    signal.signal(signal.SIGWINCH, on_resize)

    opts = parser.parse_args(args or sys.argv[1:])
    grid_dims = opts.grid_dims or [{'rows': 4, 'cols': 4}]
    base_num = opts.base or 2
    resume = opts.resume if opts.resume is not False else False
    auto = opts.auto
    viz = opts.viz if auto else True

    grids = []
    save_state = {}
    if resume is not False and not (opts.grid_dims or opts.base):
        save_state = save.load_from_file(resume)

    score = save_state.get('score', 0)
    for grid_state in save_state.get('grids', grid_dims):
        TermTile = partial(Tile, term=term,
                           base=grid_state.pop('base', base_num))
        tiles = grid_state.pop('tiles', ())

        grid = Grid(x=0, y=1, term=term, Tile=TermTile, **grid_state)
        if tiles:
            for tile_state in tiles:
                grid.spawn_tile(**tile_state)
        else:
            grid.spawn_tile()
            grid.spawn_tile()

        game_over = game_over or len(grid.possible_moves) == 0

        grids.append(grid)

    def play(do_resize, score, game_over, auto, viz, term_too_small, grid=None, agent=None):
        tot_choose_time = 0
        while True:
            if do_resize and viz:
                term_too_small = not term_resize(term, grids)
                do_resize = False

            if not term_too_small and viz:
                draw_score(score, term, end=game_over)

            # TODO move or remove

            if auto:
                # generate valid_moves array
                invalid_moves = []
                valid_moves = {}
                for grid in grids:
                    for action_encoding in range(4):
                        direction = grid_moves.get(action_encoding)
                        if len(grid.move(direction, apply=False)) == 0:
                            invalid_moves.append(action_encoding)
                        else: 
                            valid_moves[action_encoding] = []
                if len(invalid_moves) == 4: 
                    game_over = True

                afterstates = []
                for grid in grids:
                    for action_encoding in valid_moves.keys():
                        direction = grid_moves.get(action_encoding)
                        board_copy = Grid()
                        grid_copy = [[None for _ in range(4)] for _ in range(4)]
                        for row_index, row in enumerate(grid._grid):
                            for col_index, tile_thing in enumerate(row):
                                if tile_thing is not None:
                                    grid_copy[row_index][col_index] = Tile(base=tile_thing.base, exponent=tile_thing.exponent)
                        board_copy._grid = grid_copy
                        actions = board_copy.move(direction)
                        reward = 0
                        for action in actions:
                            if action.type == Actions.merge:
                                row, column = action.new
                                reward += board_copy[row][column].value
                        afterstates.append(board_copy._grid)
                        for row_index, row in enumerate(grid_copy):
                            for col_index, tile_thing in enumerate(row):
                                new_grid_copy = [[None for _ in range(4)] for _ in range(4)]
                                new_grid_copy_2 = [[None for _ in range(4)] for _ in range(4)]
                                if grid_copy[row_index][col_index] is None:
                                    new_grid_copy[row_index][col_index] = Tile(base=2, exponent=1)
                                    new_grid_copy_2[row_index][col_index] = Tile(base=2, exponent=2)
                                    valid_moves[action_encoding].append((reward,new_grid_copy,0.9))
                                    valid_moves[action_encoding].append((reward,new_grid_copy_2,0.1))
                                    
                if len(invalid_moves) == 4: 
                    game_over = True

                else:
                    # choose_time = time()      
                    # vectorize state
                    # state_representation = simple_exponent_state_rep(grid._grid)
                    state_representation = agent.state_representation_function(grid._grid)
                    # agent chooses direction based on the state
                    # chosen_action = agent.choose(state_representation, invalid_moves)
                    chosen_action = agent.choose(state=state_representation, valid_moves=valid_moves)
                    # tot_choose_time += (time()-choose_time)
                    direction = grid_moves.get(chosen_action)
            
                if game_over:
                    # agent updates behavior if episodic
                    if agent.type == 'offline': 
                        update_time = time()
                        mse = agent.update()
                        # print(f'Total Choose time = {tot_choose_time:.8f}s')
                        # print(f'time to update = {(time() - update_time):.8f}s')
                        save.write_to_file(score, grids, filename=resume or None)
                        return score, mse

                    elif agent.type == 'evolutionary':
                        save.write_to_file(score, grids, filename=resume or None)
                        return score

            else:
                key = term.inkey()
                if key in ('q', 'KEY_ESCAPE') or game_over:
                    save.write_to_file(score, grids, filename=resume or None)
                    break
                direction = grid_moves.get(key.name or key)

            if not direction or term_too_small:
                continue

            for grid in grids:
                reward = 0
                actions = grid.move(direction)

                for action in actions:
                    grid.draw_empty_tile(*action.old)

                    if action.type == Actions.merge:
                        row, column = action.new
                        reward += grid[row][column].value
                        score += grid[row][column].value
                
                if actions:  # had any successfull move(s)?
                    grid.spawn_tile(exponent=2 if random.random() > 0.9 else 1)

                    if viz or not auto: grid.draw_tiles()
                    if auto and agent.type != 'evolutionary':
                        agent.state_history.append(state_representation)
                        agent.action_history.append(chosen_action)
                        agent.reward_history.append(reward)

                if all(chain(*grid)):
                    game_over = game_over or len(grid.possible_moves) == 0
                    
    if auto:
        if viz:
            with term.fullscreen(), term.cbreak(), term.hidden_cursor():
                final_score = play(do_resize=do_resize, score=score, game_over=game_over, auto=auto, term_too_small=term_too_small, grid=grid, agent=agent, viz=viz)
        else:
            final_score, mse = play(do_resize=do_resize, score=score, game_over=game_over, auto=auto, term_too_small=term_too_small, grid=grid, agent=agent, viz=viz)
    elif not auto:
        with term.fullscreen(), term.cbreak(), term.hidden_cursor():
            final_score = play(do_resize=do_resize, score=score, game_over=game_over, auto=auto, term_too_small=term_too_small, viz=True)

    high = 0
    for max_tile in filter(None, (g.highest_tile for g in grids)):
        # high = max(high, max_tile.value)
        high = max(high, max_tile.exponent)
    if not auto: print("highest tile: {}\nscore: {}".format(high, score))

    if auto: return final_score, high, int(mse)
    return 0
