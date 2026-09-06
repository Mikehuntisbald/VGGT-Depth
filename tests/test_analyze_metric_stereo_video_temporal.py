from __future__ import annotations

import torch

from tools.analyze_metric_stereo_video_temporal import _finite_positive


def test_temporal_analysis_finite_positive_domain() -> None:
    value = torch.tensor([[[[1.0, 0.0, float("nan"), -1.0]]]])
    assert _finite_positive(value).tolist() == [[[[True, False, False, False]]]]
