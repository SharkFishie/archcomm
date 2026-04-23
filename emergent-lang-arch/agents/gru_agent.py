import torch
import torch.nn as nn
import egg.core as core


class GRUSender(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, max_len, embed_dim=64):
        super().__init__()
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, aux_input=None):
        h = torch.tanh(self.fc_input(x))   # (B, hidden_dim) — GRUCell needs 2D
        return h


class GRUReceiver(nn.Module):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__()
        self.fc_obj = nn.Linear(input_dim, hidden_dim)

    def forward(self, hidden, input, aux_input=None):
        obj_repr = torch.relu(self.fc_obj(input))
        scores = torch.bmm(obj_repr, hidden.unsqueeze(-1)).squeeze(-1)
        return scores


def build_gru_agents(cfg):
    sender_core = GRUSender(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        vocab_size=cfg["vocab_size"],
        max_len=cfg["max_len"],
    )
    receiver_core = GRUReceiver(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
    )

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
