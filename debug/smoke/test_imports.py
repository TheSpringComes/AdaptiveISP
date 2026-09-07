"""smoke: imports + registry population.

If this passes, all V1 modules load and the operator registry is
populated. Anything else in smoke/ can rely on this.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def test_imports() -> None:
    import tasks  # noqa: F401  bootstraps yolov3 sys.path

    from isp.base import ISPOperator, ParameterSpec, tanh_range, rgb2lum, rgb2hsv, hsv2rgb, lerp
    from isp.registry import OPERATORS, register, build_operator, CANONICAL_ORDER
    from isp.operators import exposure, gamma, ccm, sharpen, denoise, tone, contrast, saturation, wnb, whitebalance   # noqa: F401
    from isp.learned.samsung_modular import denoise as n_denoise, awb as n_awb, gain as n_gain, gtm as n_gtm, chroma as n_chroma, gamma as n_gamma, detail as n_detail   # noqa: F401
    from isp.operators.infinite_isp import awb as inf_awb, gain as inf_gain, contrast as inf_contrast, sharpen as inf_sharpen, denoise as inf_denoise, saturation as inf_saturation   # noqa: F401

    from pipeline import PipelineState, ISPAction, PipelineExecutor, pipeline_state_from_replay, pipeline_state_to_replay   # noqa: F401
    from search import SearchSpace, ConstraintResult   # noqa: F401
    from controller import Controller, ControllerOutput   # noqa: F401
    from controller.adaptiveisp import AdaptiveISPController, AdaptiveISPValueNet, FeatureExtractor, AdaptiveISPReward, Reward, RewardBreakdown   # noqa: F401
    from tasks.base import Task, TaskMetrics   # noqa: F401
    from tasks.detection.implementations.yolov3 import YOLOv3Detection   # noqa: F401
    from engine.trainer import Trainer   # noqa: F401
    from engine.util import set_seed   # noqa: F401

    assert set(CANONICAL_ORDER) == set(OPERATORS.keys()), \
        f"registry mismatch: CANONICAL_ORDER={CANONICAL_ORDER} vs OPERATORS={sorted(OPERATORS.keys())}"
    # 10 classical + 7 neural (Samsung, V2-AI) + 9 infinite-isp (V2-AI).
    assert len(OPERATORS) == 26, f"expected 26 operators, got {len(OPERATORS)}"


if __name__ == "__main__":
    test_imports()
    print("smoke/test_imports: PASS")
