REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_agent_rmix import RmixAgent

REGISTRY["rnn_agent"] = RNNAgent
REGISTRY["rnn_agent_rmix"] = RmixAgent
