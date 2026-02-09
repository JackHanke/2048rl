import math
import torch
import torch.nn as nn
from torchsummary import summary


def sinusoidal_positional_encoding(seq_len, d_model, n=10000.0):
    positions = torch.arange(0, seq_len).unsqueeze(1)  # [seq_len, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(n) / d_model))  # [d_model//2]
    
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    
    return pe  # Shape: [seq_len, d_model]   

# double headed policy value network
class PolicyValueNet(nn.Module):
    def __init__(self, 
            embedding_dim: int,
            num_layers: int,
            device,
        ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # NOTE class token is embedding 18
        self.embeddings = torch.nn.Embedding(num_embeddings=18, embedding_dim=embedding_dim)

        # self.positional_encoding = nn.Parameter(torch.zeros(17, self.embedding_dim))
        # nn.init.xavier_uniform_(self.positional_encoding)  # Initialize with Xavier uniform

        self.positional_encoding = sinusoidal_positional_encoding(seq_len=17, d_model=self.embedding_dim).to(device)

        self.torso = nn.Sequential(*[
            nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=16,
            dim_feedforward=self.embedding_dim,
            batch_first=True
        ) for _ in range(self.num_layers)])

        self.policy_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim//2, 4),
        )

        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim//2, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.tensor):
        x = self.embeddings(x)

        x = x + self.positional_encoding
        
        x = self.torso(x)

        policy_logits = self.policy_head(x[:, 0])

        value = self.value_head(x[:, 0])

        return policy_logits, value

class PolicyNet(nn.Module):
    def __init__(self, 
            embedding_dim: int,
            num_layers: int,
            device,
        ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # NOTE class token is embedding 18
        self.embeddings = torch.nn.Embedding(num_embeddings=18, embedding_dim=embedding_dim)

        # self.positional_encoding = nn.Parameter(torch.zeros(17, self.embedding_dim))
        # nn.init.xavier_uniform_(self.positional_encoding)  # Initialize with Xavier uniform
        self.positional_encoding = sinusoidal_positional_encoding(seq_len=17, d_model=self.embedding_dim).to(device)

        self.torso = nn.Sequential(*[
            nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=16,
            dim_feedforward=self.embedding_dim,
            batch_first=True
        ) for _ in range(self.num_layers)])

        self.policy_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim//2, 4),
        )

    def forward(self, x: torch.tensor):
        x = self.embeddings(x)

        x = x + self.positional_encoding
        
        x = self.torso(x)

        policy_logits = self.policy_head(x[:, 0])

        return policy_logits

if __name__ == '__main__':
    import yaml
    with open('config.yaml', 'r') as file: config = yaml.safe_load(file)
    EMBEDDING_DIM = config['embedding_dim']
    NUM_LAYERS = config['num_layers']
    BATCH_SIZE = config['batch_size']

    model = PolicyValueNet(
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
    )
    model.eval()
    
    state = torch.randint(low=0, high=17, size=(BATCH_SIZE, 17))
    logits, value = model(state)

    summary(model, input_size=state.shape)

    print(f'logits: {logits}')
    print(f'logits.shape: {logits.shape}')
    print(f'value: {value}')
    print(f'value.shape: {value.shape}')
