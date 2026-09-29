import pytest
import torch

from neural_style_transfer.device import get_best_device


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_fallback_order(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert get_best_device().type == expected
    assert get_best_device("auto").type == expected


def test_explicit_cpu_is_always_allowed():
    assert get_best_device("cpu") == torch.device("cpu")


def test_unavailable_device_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        get_best_device("cuda")


def test_selected_device_runs_a_kernel():
    device = get_best_device()
    x = torch.randn(4, 4, device=device)
    assert torch.allclose((x @ x.T).cpu(), (x.cpu() @ x.cpu().T), atol=1e-4)
