from tqdm import tqdm
from game.gameof2048 import Gameof2048

def batch_eval(
        agent,
        batch_size,
        doing_rl_training: bool = False
    ):
    ''' Evaluate the agent by playing a batch of simultaneous games '''
    games = [Gameof2048(game_idx=game_idx) for game_idx in range(batch_size)]

    prog_bar = tqdm(total=batch_size)

    moves_played = 0
    all_games_over = False
    while not all_games_over:
        # make state
        boards = [game.board for game in games if not game.game_over]
        if len(boards) == 0: 
            all_games_over = True
        else:
            actions, logits, values = agent.choose(boards=boards, return_logits=True)

            # make move
            schedule = [game for game in games if not game.game_over]
            for playing_game_idx, game in enumerate(schedule):
                action = actions[playing_game_idx].item()
                reward = game.do_move(action=action)
                moves_played += 1

                # 
                if doing_rl_training:
                    agent.add(
                        board=game.board,
                        action=action,
                        reward=reward,
                        logits=logits[playing_game_idx],
                        value=values[playing_game_idx], 
                        game_idx=game.game_idx,
                    )
                    # if game.do_move ended the game, finish trajectory
                    if game.game_over:
                        agent.buffer.finish_trajectory(game_idx=game.game_idx)
                        
                if game.game_over:
                    prog_bar.update(1)

    scores = [int(game.board.score) for game in games]
    return scores


if __name__ == '__main__':
    import torch
    from nets.agent import Agent

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    agent = Agent(
        ply=0,
        games_per_iter=12, 
        device=DEVICE,
        mode='training'
    )

    batch_eval(
        agent=agent,

    )


