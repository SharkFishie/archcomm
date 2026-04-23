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
    def __init__(self, input_dim, hidden_dim, output_dim, embed_dim=64, vocab_size=50):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_obj = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, message, input, aux_input=None):
        # message: (B, L) token ids
        emb = self.embed(message.long())
        _, (h, _) = self.lstm(emb)
        h = h.squeeze(0)

        # input: (B, n_candidates, input_dim)
        obj_repr = torch.relu(self.fc_obj(input))   # (B, n_candidates, hidden_dim)
        scores = torch.bmm(obj_repr, h.unsqueeze(-1)).squeeze(-1)  # (B, n_candidates)
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
        output_dim=cfg["n_distractors"] + 1,
        vocab_size=cfg["vocab_size"],
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
