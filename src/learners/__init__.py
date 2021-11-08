from .q_learner import QLearner
from .rmix_learner import RMIXLearner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["rmix_learner"] = RMIXLearner
