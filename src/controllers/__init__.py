REGISTRY = {}

from .basic_controller import BasicMAC
from .rmix_controller import RmixMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["risk_mac"] = RmixMAC
