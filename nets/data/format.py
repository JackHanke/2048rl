# make single large csv from: https://huggingface.co/datasets/DeeperGame/game-2048/blob/main/README.md
import csv
import os
from math import log
from tqdm import tqdm

map_move_to_num = {
    '↑': 0,
    '→': 1,
    '↓': 2,
    '←': 3,
}

def modified_log(integer: int):
    if integer == 0: return 0
    return int(log(integer, 2))

def process_board_row(board_row):
    return [modified_log(int(i.strip())) for i in board_row]

if __name__ == '__main__':
    WRITE_TO_PATH = f'./nets/data/data.csv'
    DATA_DIR = f'./nets/data/games_1570'
    with open(WRITE_TO_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['board_state', 'move'])

        # input('check')
        prog_bar = tqdm(sorted(os.listdir(DATA_DIR)))
        for file in prog_bar:
            with open(f'./nets/data/games_1570/{file}', 'r') as csv_file:
                reader = csv.reader(csv_file)
                for row_idx, row in enumerate(reader):
                    if (row_idx % 5) == 0:
                        board = process_board_row(row)
                    elif (row_idx % 5) == 4:
                        if row[3].strip() != '_':
                            row_to_write = [f'"{board}"', map_move_to_num[row[3].strip()]]
                            writer.writerow(row_to_write)
                    else:
                        board.extend(process_board_row(row))
            prog_bar.set_description(f'Read: {file}')



