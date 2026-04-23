import torch
import torch.nn as nn
import egg.core as core


class MLPSender(nn.Module):
    """Feedforward sender — no recurrence, encodes object into hidden state."""

    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, aux_input=None):
        h = torch.tanh(self.net(x))   # (B, hidden_dim) — GRUCell needs 2D
        return h


class MLPReceiver(nn.Module):
    """Receiver that scores candidates against EGG's encoded message hidden state."""

    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__()
        self.fc_obj = nn.Linear(input_dim, hidden_dim)

    def forward(self, hidden, input, aux_input=None):
        obj_repr = torch.relu(self.fc_obj(input))
        scores = torch.bmm(obj_repr, hidden.unsqueeze(-1)).squeeze(-1)
        return scores


def build_mlp_agents(cfg):
    sender_core = MLPSender(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg.get("mlp_layers", 2),
    )
    receiver_core = MLPReceiver(
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
