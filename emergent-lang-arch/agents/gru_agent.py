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
    def __init__(self, input_dim, hidden_dim, output_dim, embed_dim=64, vocab_size=50):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_obj = nn.Linear(input_dim, hidden_dim)

    def forward(self, message, input, aux_input=None):
        emb = self.embed(message.long())
        _, h = self.gru(emb)
        h = h.squeeze(0)

        obj_repr = torch.relu(self.fc_obj(input))
        scores = torch.bmm(obj_repr, h.unsqueeze(-1)).squeeze(-1)
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
        output_dim=cfg["n_distractors"] + 1,
        vocab_size=cfg["vocab_size"],
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
