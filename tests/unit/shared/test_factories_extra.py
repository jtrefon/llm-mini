"""Extra tests to increase coverage for optimizer factory internals."""
import torch
import torch.nn as nn

from src.shared.factories.optimizer_factory import _coerce_parameters


def test_coerce_parameters_empty_iterable():
    # Empty iterable should return empty list (handled upstream by callers)
    result = _coerce_parameters([])
    assert isinstance(result, list)
    assert result == []


def test_coerce_parameters_with_tensor_param():
    # Parameter is iterable; function peeks first element and returns tensors list
    p = nn.Parameter(torch.zeros(1, requires_grad=True))
    result = _coerce_parameters(p)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], (nn.Parameter, torch.Tensor))


def test_coerce_parameters_with_invalid_non_iterable():
    # Invalid non-iterable should fall back to dummy parameter
    result = _coerce_parameters(object())
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], nn.Parameter)


def test_coerce_parameters_scalar_parameter_hits_typeerror_branch():
    # Zero-dim Parameter is non-iterable and should be wrapped as-is
    scalar = nn.Parameter(torch.tensor(0.0, requires_grad=True))
    result = _coerce_parameters(scalar)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], nn.Parameter)
