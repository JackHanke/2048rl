import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from math import log

logger = logging.getLogger(__name__)

def preprocess_board_from_csv(board_str):
    board_str = board_str[2:-2] # remove quotes and brackets, very hacky
    # NOTE class token first
    state_tensor = torch.tensor([17]+[int(tile_str.strip()) for tile_str in board_str.split(',')], dtype=torch.int)
    return state_tensor

class PretrainingSet(Dataset):
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = preprocess_board_from_csv(board_str=row.iloc[0])
        label = torch.tensor(row.iloc[1], dtype=torch.long)
        return features, label

