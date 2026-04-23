import torch
import torch.nn as nn
import math
import egg.core as core


class TransformerSender(nn.Module):
    """Sender that encodes the target object and produces initial hidden state."""

    def __init__(self, input_dim, hidden_dim, nhead=4, num_layers=2):
        super().__init__()
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden_dim = hidden_dim

    def forward(self, x, aux_input=None):
        # x: (B, input_dim) — single object vector
        h = torch.relu(self.fc_input(x)).unsqueeze(1)   # (B, 1, hidden_dim)
        h = self.transformer(h).squeeze(1)               # (B, hidden_dim) — LSTMCell/GRUCell need 2D
        return h


class TransformerReceiver(nn.Module):
    """Receiver that scores candidate objects against EGG's encoded message hidden state."""

    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__()
        self.fc_obj = nn.Linear(input_dim, hidden_dim)

    def forward(self, hidden, input, aux_input=None):
        obj_repr = torch.relu(self.fc_obj(input))
        scores = torch.bmm(obj_repr, hidden.unsqueeze(-1)).squeeze(-1)
        return scores


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def build_transformer_agents(cfg):
    sender_core = TransformerSender(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 2),
    )
    receiver_core = TransformerReceiver(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
    )

    # Wrap with EGG RNN shells — transformer hidden state seeds the GRU decoder
    sender = core.RnnSenderReinforce(
        sender_core,
        vocab_size=cfg["vocab_size"],
        embed_dim=64,
        hidden_size=cfg["hidden_dim"],
        max_len=cfg["max_len"],
        cell="gru",
    )
    receiver = core.RnnReceiverDeterministic(
        receiver_core,
        vocab_size=cfg["vocab_size"],
        embed_dim=64,
        hidden_size=cfg["hidden_dim"],
        cell="gru",
    )
    return sender, receiver
