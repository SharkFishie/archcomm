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
    """Feedforward receiver — encodes message mean-pooled embedding + object repr."""

    def __init__(self, input_dim, hidden_dim, vocab_size, embed_dim=64, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

        layers = []
        in_dim = embed_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        self.msg_net = nn.Sequential(*layers)

        self.fc_obj = nn.Linear(input_dim, hidden_dim)

    def forward(self, message, input, aux_input=None):
        # Mean-pool token embeddings (no ordering info — purely ablation baseline)
        emb = self.embed(message.long()).mean(dim=1)            # (B, embed_dim)
        h = torch.relu(self.msg_net(emb))                # (B, hidden_dim)

        obj_repr = torch.relu(self.fc_obj(input))        # (B, n_candidates, hidden_dim)
        scores = torch.bmm(obj_repr, h.unsqueeze(-1)).squeeze(-1)
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
        vocab_size=cfg["vocab_size"],
        num_layers=cfg.get("mlp_layers", 2),
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
