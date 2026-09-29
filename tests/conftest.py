import numpy as np
import pytest
from PIL import Image


def make_image(path, size=(96, 64), seed=0):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def style_path(tmp_path):
    return make_image(tmp_path / "style.png", size=(80, 80), seed=1)


@pytest.fixture
def content_paths(tmp_path):
    folder = tmp_path / "content"
    folder.mkdir()
    return [make_image(folder / "wide.png", (96, 64), 2), make_image(folder / "tall.jpg", (48, 72), 3)]
