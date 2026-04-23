"""
Referential game dataset and loss.

Protocol:
  - Sender sees a target object feature vector.
  - Receiver sees (n_distractors + 1) candidate vectors in shuffled order
    and must identify which is the target using the sender's message.
  - Loss: cross-entropy over candidate positions (receiver output).
  - Reward for REINFORCE: 1.0 if receiver correct, 0.0 otherwise.
"""

import torch
from torch.utils.data import Dataset
import egg.core as core


class ReferentialDataset(Dataset):
    def __init__(self, n_objects: int, n_features: int, n_distractors: int, n_samples: int, seed: int = 42):
        rng = torch.Generator()
        rng.manual_seed(seed)

        # Each sample: one target drawn from n_objects prototypes
        prototypes = torch.rand(n_objects, n_features, generator=rng)
        self.prototypes = prototypes

        indices = torch.randint(0, n_objects, (n_samples,), generator=rng)
        self.targets = prototypes[indices]                 # (N, n_features)
        self.labels = torch.zeros(n_samples, dtype=torch.long)  # target always at position 0 before shuffle

        # Pre-sample distractor indices (different from target)
        distractor_idx = torch.zeros(n_samples, n_distractors, dtype=torch.long)
        for i in range(n_samples):
            pool = list(range(n_objects))
            pool.remove(int(indices[i]))
            chosen = torch.randperm(len(pool), generator=rng)[:n_distractors]
            distractor_idx[i] = torch.tensor([pool[c] for c in chosen])

        distractors = prototypes[distractor_idx]          # (N, n_distractors, n_features)

        # Build candidate set: [target, d1, d2, ...] then shuffle per sample
        all_candidates = torch.cat([self.targets.unsqueeze(1), distractors], dim=1)  # (N, K, F)

        perm = torch.stack([torch.randperm(n_distractors + 1, generator=rng) for _ in range(n_samples)])
        self.candidates = torch.stack([all_candidates[i][perm[i]] for i in range(n_samples)])

        # Label = position of target after shuffle
        self.labels = torch.tensor([(perm[i] == 0).nonzero(as_tuple=True)[0].item() for i in range(n_samples)])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # EGG expects: sender_input, labels, receiver_input
        return self.targets[idx], self.labels[idx], self.candidates[idx]


def referential_loss(sender_input, message, receiver_input, receiver_output, labels, aux_input=None):
    """Cross-entropy loss + accuracy for the referential game."""
    loss = torch.nn.functional.cross_entropy(receiver_output, labels, reduction="none")
    acc = (receiver_output.argmax(dim=-1) == labels).float()
    return loss, {"acc": acc}


def build_game(sender, receiver, cfg):
    loss_fn = referential_loss
    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss_fn,
        sender_entropy_coeff=cfg.get("sender_entropy_coeff", 0.01),
        receiver_entropy_coeff=cfg.get("receiver_entropy_coeff", 0.001),
    )
    return game
