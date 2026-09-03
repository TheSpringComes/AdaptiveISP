"""controller: the algorithm research core.

Answers "how does the framework choose the next ISP action?"

V1 has one concrete family: `controller.adaptiveisp` (RL policy + value net,
ported from the original AdaptiveISP paper). Future families would sit as
sibling subpackages: `controller.bayesopt`, `controller.cmaes`, ...
"""
from controller.base import Controller, ControllerOutput

__all__ = ["Controller", "ControllerOutput"]
