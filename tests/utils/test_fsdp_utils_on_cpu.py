import torch.nn as nn
import pytest

from verl.utils.fsdp_utils import get_fsdp_wrap_policy


class PresentLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)


class ToyModel(nn.Module):
    _no_split_modules = ["PresentLayer", "MissingAlternateLayer"]

    def __init__(self) -> None:
        super().__init__()
        self.layer = PresentLayer()


class NoMatchModel(nn.Module):
    _no_split_modules = ["MissingLayer"]

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(4, 4)


def test_get_fsdp_wrap_policy_allows_missing_alternative_layer_classes() -> None:
    model = ToyModel()

    with pytest.warns(UserWarning, match="will be ignored"):
        auto_wrap_policy = get_fsdp_wrap_policy(model)

    assert auto_wrap_policy is not None


def test_get_fsdp_wrap_policy_raises_when_no_requested_layer_exists() -> None:
    model = NoMatchModel()

    with pytest.raises(Exception, match="Could not find any transformer layer classes"):
        get_fsdp_wrap_policy(model)
