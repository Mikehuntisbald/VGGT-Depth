import pytest

torch = pytest.importorskip("torch")

from models import TopKHistoryEncoder, TopKHistoryEncoding


def _inputs(
    *, batch: int = 2, candidates: int = 3, height: int = 3, width: int = 4
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    scalar_shape = (batch, candidates, height, width)
    return {
        "disparity_hr_px": 1.0 + 8.0 * torch.rand(
            scalar_shape, generator=generator
        ),
        "confidence": torch.rand(scalar_shape, generator=generator),
        "fractional_phase_grid_px": 2.0
        * torch.rand((batch, candidates, 2, height, width), generator=generator)
        - 1.0,
        "temporal_age_frames": torch.randint(
            1, 4, scalar_shape, generator=generator
        ).float(),
        "z_aware_weights": torch.rand(scalar_shape, generator=generator),
        "valid_mask": torch.rand(scalar_shape, generator=generator) > 0.25,
    }


def test_shapes_and_explicit_weighted_aggregation() -> None:
    model = TopKHistoryEncoder()
    inputs = _inputs()
    output = model(**inputs)

    assert isinstance(output, TopKHistoryEncoding)
    assert output.candidate_feature.shape == (2, 3, 32, 3, 4)
    assert output.aggregate_feature.shape == (2, 32, 3, 4)
    assert output.effective_weights.shape == (2, 3, 3, 4)
    assert output.aggregate_valid_mask.shape == (2, 1, 3, 4)
    expected = (
        output.effective_weights.unsqueeze(2) * output.candidate_feature
    ).sum(dim=1)
    torch.testing.assert_close(output.aggregate_feature, expected)
    torch.testing.assert_close(
        output.effective_weights.sum(dim=1, keepdim=True)[
            output.aggregate_valid_mask
        ],
        torch.ones_like(
            output.effective_weights.sum(dim=1, keepdim=True)[
                output.aggregate_valid_mask
            ]
        ),
    )


def test_invalid_candidates_are_exact_zero_and_receive_no_weight() -> None:
    model = TopKHistoryEncoder()
    inputs = _inputs(batch=1, candidates=3, height=2, width=2)
    inputs["valid_mask"][:] = False
    inputs["disparity_hr_px"][:] = float("nan")
    inputs["confidence"][:] = float("inf")
    inputs["fractional_phase_grid_px"][:] = float("-inf")
    inputs["temporal_age_frames"][:] = float("nan")
    inputs["z_aware_weights"][:] = float("inf")
    # Learned affine offsets must not leak through the invalid mask.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.GroupNorm):
                module.bias.fill_(3.0)

    output = model(**inputs)

    assert output.candidate_feature.eq(0).all()
    assert output.aggregate_feature.eq(0).all()
    assert output.effective_weights.eq(0).all()
    assert not bool(output.aggregate_valid_mask.any())
    assert bool(torch.isfinite(output.aggregate_feature).all())


def test_nonfinite_valid_inputs_are_sanitized_before_encoding() -> None:
    model = TopKHistoryEncoder(maximum_disparity_hr_px=100.0, maximum_age_frames=5.0)
    inputs = _inputs(batch=1, candidates=2, height=2, width=2)
    inputs["valid_mask"][:] = True
    inputs["disparity_hr_px"][0, 0, 0, 0] = float("inf")
    inputs["confidence"][0, 0, 0, 1] = float("nan")
    inputs["fractional_phase_grid_px"][0, 1, 0, 1, 0] = float("inf")
    inputs["temporal_age_frames"][0, 1, 1, 1] = float("-inf")
    inputs["z_aware_weights"][0, 1, 0, 0] = float("nan")

    output = model(**inputs)

    assert bool(torch.isfinite(output.candidate_feature).all())
    assert bool(torch.isfinite(output.aggregate_feature).all())
    assert bool(torch.isfinite(output.effective_weights).all())


def test_weights_are_valid_masked_and_renormalized_without_uniform_fallback() -> None:
    model = TopKHistoryEncoder()
    inputs = _inputs(batch=1, candidates=3, height=1, width=1)
    inputs["z_aware_weights"] = torch.tensor([[[[0.2]], [[0.3]], [[0.5]]]])
    inputs["valid_mask"] = torch.tensor([[[[True]], [[False]], [[True]]]])

    output = model(**inputs)

    torch.testing.assert_close(
        output.effective_weights.flatten(), torch.tensor([2.0 / 7.0, 0.0, 5.0 / 7.0])
    )
    inputs["z_aware_weights"].zero_()
    zero_weight = model(**inputs)
    assert not bool(zero_weight.aggregate_valid_mask.any())
    assert zero_weight.effective_weights.eq(0).all()
    assert zero_weight.aggregate_feature.eq(0).all()


def test_shared_candidate_encoder_is_permutation_equivariant() -> None:
    model = TopKHistoryEncoder().eval()
    inputs = _inputs(batch=1, candidates=4, height=2, width=3)
    original = model(**inputs)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted_inputs = {
        name: (
            value[:, permutation]
            if name != "fractional_phase_grid_px"
            else value[:, permutation]
        )
        for name, value in inputs.items()
    }
    permuted = model(**permuted_inputs)

    torch.testing.assert_close(
        permuted.candidate_feature, original.candidate_feature[:, permutation]
    )
    torch.testing.assert_close(
        permuted.effective_weights, original.effective_weights[:, permutation]
    )
    torch.testing.assert_close(permuted.aggregate_feature, original.aggregate_feature)


def test_runtime_k_does_not_change_parameters_or_state_dict() -> None:
    model = TopKHistoryEncoder()
    state_keys = tuple(model.state_dict())
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())

    for candidates in (1, 2, 7):
        output = model(
            **_inputs(batch=1, candidates=candidates, height=2, width=2)
        )
        assert output.candidate_feature.shape[1] == candidates
        assert tuple(model.state_dict()) == state_keys
        assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids


def test_input_contract_rejects_ambiguous_shapes_and_masks() -> None:
    model = TopKHistoryEncoder()
    inputs = _inputs(batch=1, candidates=2, height=2, width=2)
    bad_phase = dict(inputs)
    bad_phase["fractional_phase_grid_px"] = torch.zeros((1, 2, 1, 2, 2))
    with pytest.raises(ValueError, match="fractional_phase"):
        model(**bad_phase)
    bad_mask = dict(inputs)
    bad_mask["valid_mask"] = inputs["valid_mask"].float()
    with pytest.raises(TypeError, match="bool"):
        model(**bad_mask)
    with pytest.raises(ValueError, match="output_channels"):
        TopKHistoryEncoder(output_channels=0)
