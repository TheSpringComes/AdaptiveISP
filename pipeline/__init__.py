"""pipeline: pipeline state + action + executor.

Answers "how the ISP pipeline advances one step at a time":
    PipelineExecutor.step(state, action) -> state'

CanonicalBackbone 已迁移到 `front_isp.canonical`（Front ISP 现为独立
可插拔模块，见 front_isp/）。本包只保留 rollout 运行时组件。
"""
from pipeline.action import ISPAction
from pipeline.executor import PipelineExecutor, pipeline_state_from_replay, pipeline_state_to_replay
from pipeline.state import PipelineState
from pipeline.trajectory import TrajectoryBuffer

__all__ = [
    "PipelineState", "ISPAction", "PipelineExecutor",
    "TrajectoryBuffer",
    "pipeline_state_from_replay", "pipeline_state_to_replay",
]
