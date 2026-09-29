import pytest
import torch
from torch import nn

from neural_style_transfer.model import CONTENT_LAYERS, STYLE_LAYERS, VGGFeatures, gram_matrix


def test_gram_matrix_matches_manual_computation():
    x = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    flat = x.reshape(2, 3, 4)
    expected = torch.stack([f @ f.T for f in flat]) / (3 * 2 * 2)
    assert torch.allclose(gram_matrix(x), expected)


def test_gram_matrix_is_resolution_independent():
    # Tiling a feature map 2x2 keeps its channel statistics, so the Gram matrix should not change.
    x = torch.rand(1, 8, 5, 7)
    assert torch.allclose(gram_matrix(x), gram_matrix(x.repeat(1, 1, 2, 2)), atol=1e-6)


@pytest.fixture(scope="module")
def vgg():
    return VGGFeatures(pretrained=False)


def test_returns_requested_layers_with_expected_channels(vgg):
    feats = vgg(torch.rand(1, 3, 64, 64))
    assert set(feats) == set(STYLE_LAYERS + CONTENT_LAYERS)
    channels = {name: f.shape[1] for name, f in feats.items()}
    assert channels == {"relu1_1": 64, "relu2_1": 128, "relu3_1": 256, "relu4_1": 512, "relu4_2": 512, "relu5_1": 512}
    assert all((f >= 0).all() for f in feats.values()), "activations should be post-ReLU"


def test_network_is_frozen_and_truncated(vgg):
    assert not any(p.requires_grad for p in vgg.parameters())
    assert len(vgg.body) == 30  # nothing after relu5_1
    assert not any(m.inplace for m in vgg.body if isinstance(m, nn.ReLU))


def test_unknown_layer_is_rejected():
    with pytest.raises(ValueError, match="conv9_9"):
        VGGFeatures(["conv9_9"], pretrained=False)
