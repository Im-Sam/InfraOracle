import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):

    def __init__(self, n_features, hidden_size=64, embedding_dim=32):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.embedding = nn.Linear(hidden_size, embedding_dim)

        self.decoder_input = nn.Linear(embedding_dim, hidden_size)

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            num_layers=2,
            batch_first=True
        )

    def forward(self, x):

        _, (hidden, _) = self.encoder(x)

        latent = self.embedding(hidden[-1])

        repeated = latent.unsqueeze(1).repeat(1, x.shape[1], 1)

        decoded_input = self.decoder_input(repeated)

        reconstructed, _ = self.decoder(decoded_input)

        return reconstructed