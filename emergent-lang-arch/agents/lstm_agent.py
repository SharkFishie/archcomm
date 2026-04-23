import torch
import torch.nn as nn
import egg.core as core


class LSTMSender(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, max_len, embed_dim=64):
        super().__init__()
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.vocab_size = vocab_size

    def forward(self, x, aux_input=None):
        h = torch.tanh(self.fc_input(x))   # (B, hidden_dim) — LSTMCell needs 2D
        c = torch.zeros_like(h)
        return h, c


class LSTMReceiver(nn.Module):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__()
        # EGG's RnnReceiverDeterministic runs its own LSTM encoder and passes
        # the final hidden state here — no re-encoding needed.
        self.fc_obj = nn.Linear(input_dim, hidden_dim)

    def forward(self, hidden, input, aux_input=None):
        # hidden: (B, hidden_dim) already encoded by EGG's RnnEncoder
        # input:  (B, n_candidates, input_dim)
        obj_repr = torch.relu(self.fc_obj(input))                    # (B, K, hidden_dim)
        scores = torch.bmm(obj_repr, hidden.unsqueeze(-1)).squeeze(-1)  # (B, K)
        return scores


def build_lstm_agents(cfg):
    sender_core = LSTMSender(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        vocab_size=cfg["vocab_size"],
        max_len=cfg["max_len"],
    )
    receiver_core = LSTMReceiver(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
    )

    sender = core.RnnSenderReinforce(
        sender_core,
        vocab_size=cfg["vocab_size"],
        embed_dim=64,
        hidden_size=cfg["hidden_dim"],
        max_len=cfg["max_len"],
        cell="lstm",
    )
    receiver = core.RnnReceiverDeterministic(
        receiver_core,
        vocab_size=cfg["vocab_size"],
        embed_dim=64,
        hidden_size=cfg["hidden_dim"],
        cell="lstm",
    )
    return sender, receiver
