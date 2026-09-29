import torch
from PIL import Image

from neural_style_transfer.transfer import StyleConfig, StyleTransfer, load_image, to_pil


def test_load_image_keeps_aspect_ratio(content_paths):
    wide, tall = content_paths
    assert load_image(wide, 48).shape == (1, 3, 32, 48)
    assert load_image(tall, 36).shape == (1, 3, 36, 24)


def test_load_image_accepts_bytes_and_pil(content_paths):
    path = content_paths[0]
    from_path = load_image(path, 48)
    assert torch.equal(load_image(path.read_bytes(), 48), from_path)
    assert torch.equal(load_image(Image.open(path), 48), from_path)


def test_to_pil_round_trip():
    x = torch.rand(1, 3, 10, 12)
    image = to_pil(x)
    assert image.size == (12, 10)
    assert torch.allclose(load_image(image, 12), x, atol=1 / 255)


def test_stylize_reduces_loss_and_keeps_shape(style_path, content_paths):
    engine = StyleTransfer(style_path, StyleConfig(size=48, steps=10), device="cpu", pretrained=False)
    result = engine.stylize(content_paths[0])
    assert result.image.size == (48, 32)
    assert result.steps == 10
    assert result.evaluations == len(result.loss_history) >= result.steps
    assert result.loss_history[-1] < result.loss_history[0]
    assert result.seconds > 0
