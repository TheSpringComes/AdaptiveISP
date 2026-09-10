"""pipeline: pipeline state + action + executor.

Answers "how the ISP pipeline advances one step at a time":
    PipelineExecutor.step(state, action) -> state'
"""
from pipeline.action import ISPAction
from pipeline.backbone import CanonicalBackbone
from pipeline.executor import PipelineExecutor, pipeline_state_from_replay, pipeline_state_to_replay
from pipeline.state import PipelineState

__all__ = [
    "PipelineState", "ISPAction", "PipelineExecutor",
    "CanonicalBackbone",
    "pipeline_state_from_replay", "pipeline_state_to_replay",
]
