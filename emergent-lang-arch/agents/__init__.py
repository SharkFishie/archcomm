import egg.core as core

from .lstm_agent import build_lstm_agents
from .gru_agent import build_gru_agents
from .transformer_agent import build_transformer_agents
from .mlp_agent import build_mlp_agents

AGENT_REGISTRY = {
    "lstm": build_lstm_agents,
    "gru": build_gru_agents,
    "transformer": build_transformer_agents,
    "mlp": build_mlp_agents,
}

# RNN cell type used inside each arch's EGG message decoder
_CELL_TYPES = {
    "lstm": "lstm",
    "gru": "gru",
    "transformer": "gru",
    "mlp": "gru",
}


def get_agents(arch: str, cfg: dict):
    if arch not in AGENT_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {list(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[arch](cfg)


def get_agents_gs(arch: str, cfg: dict):
    """
    Gumbel-Softmax variant: same sender/receiver cores as REINFORCE,
    rewrapped with RnnSenderGS + RnnReceiverGS for differentiable training.
    """
    if arch not in AGENT_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {list(AGENT_REGISTRY)}")

    # Build REINFORCE-wrapped agents just to get the cores, then discard the wrappers
    reinforce_sender, reinforce_receiver = AGENT_REGISTRY[arch](cfg)
    sender_core = reinforce_sender.agent
    receiver_core = reinforce_receiver.agent
    cell = _CELL_TYPES[arch]

    gs_sender = core.RnnSenderGS(
        sender_core,
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg.get("embed_dim", 64),
        hidden_size=cfg["hidden_dim"],
        max_len=cfg["max_len"],
        temperature=cfg.get("temperature", 1.0),
        cell=cell,
    )
    gs_receiver = core.RnnReceiverGS(
        receiver_core,
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg.get("embed_dim", 64),
        hidden_size=cfg["hidden_dim"],
        cell=cell,
    )
    return gs_sender, gs_receiver
