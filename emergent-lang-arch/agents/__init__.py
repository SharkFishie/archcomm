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


def get_agents(arch: str, cfg: dict):
    if arch not in AGENT_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {list(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[arch](cfg)
