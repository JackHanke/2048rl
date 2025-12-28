import torch
import torch.nn as nn

# 
class PolicyValueNet(nn.module):
    def __init__(self, 
            embedding_dim: int,
            num_layers: int,
        ):
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # TODO embeddings
        # TODO class token
        # TODO mlp off of class token
        # TODO 2d positional embeddings
        self.encoderlayer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=16,
            dim_feedforward=self.embedding_dim,
            batch_first=True
        )

        self.torso = nn.Sequential(*[self.encoderlayer], self.num_layers)

        self.policy_head = nn.Sequential(
            nn.Linear(),
            nn.LeakyReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.tensor):
        
        x = self.torso(x)

        policy = self.policy_head(x)
        policy_logits = policy # TODO


        value = self.value_head(x)

        return policy_logits, value
