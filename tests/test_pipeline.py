from pathlib import Path

import pytest
import ray

from neural_style_transfer.pipeline import (
    MPS_RESOURCE,
    detect_cluster_device,
    find_images,
    plan_resources,
    run_style_transfer,
)
from neural_style_transfer.transfer import StyleConfig


def test_detect_cluster_device():
    assert detect_cluster_device({"CPU": 8, "GPU": 2}) == "cuda"
    assert detect_cluster_device({"CPU": 8, MPS_RESOURCE: 1}) == "mps"
    assert detect_cluster_device({"CPU": 8}) == "cpu"


def test_plan_one_worker_per_gpu_by_default():
    plan = plan_resources("cuda", {"CPU": 32, "GPU": 4})
    assert (plan.workers, plan.num_gpus) == (4, 1.0)
    kwargs = plan.map_batches_kwargs()
    assert kwargs == {"concurrency": 4, "num_cpus": 1, "num_gpus": 1.0}


def test_plan_fractional_gpus_packs_workers():
    plan = plan_resources("cuda", {"CPU": 32, "GPU": 4}, accelerator_fraction=0.5)
    assert (plan.workers, plan.num_gpus) == (8, 0.5)


def test_plan_mps_uses_custom_resource():
    plan = plan_resources("mps", {"CPU": 16, MPS_RESOURCE: 1}, accelerator_fraction=0.5)
    assert plan.workers == 2
    assert plan.map_batches_kwargs() == {"concurrency": 2, "num_cpus": 1, "resources": {MPS_RESOURCE: 0.5}}


def test_plan_cpu_splits_cores_between_workers():
    plan = plan_resources("cpu", {"CPU": 16}, workers=3)
    assert (plan.workers, plan.num_cpus, plan.torch_threads) == (3, 5, 5)


@pytest.mark.parametrize(
    ("device", "cluster", "workers"),
    [("cuda", {"CPU": 8, "GPU": 1}, 2), ("mps", {"CPU": 8}, None), ("cpu", {"CPU": 4}, 8)],
)
def test_plan_rejects_requests_the_cluster_cannot_fit(device, cluster, workers):
    with pytest.raises(ValueError):
        plan_resources(device, cluster, workers)


def test_find_images_expands_directories(tmp_path, content_paths):
    (tmp_path / "content" / "notes.txt").write_text("not an image")
    found = find_images([tmp_path / "content"])
    assert [Path(p).name for p in found] == ["tall.jpg", "wide.png"]


def test_clashing_output_names_are_rejected(tmp_path, style_path, content_paths):
    other = tmp_path / "other"
    other.mkdir()
    (other / "wide.jpg").write_bytes(content_paths[0].read_bytes())
    with pytest.raises(ValueError, match="wide"):
        run_style_transfer([tmp_path / "content", other], style_path, output_dir=tmp_path / "out")


@pytest.fixture(scope="module")
def local_ray():
    ray.init(num_cpus=3, include_dashboard=False)
    yield
    ray.shutdown()


@pytest.mark.ray
def test_pipeline_end_to_end(local_ray, tmp_path, style_path, content_paths):
    broken = tmp_path / "content" / "broken.jpg"
    broken.write_bytes(b"not really a jpeg")

    records = run_style_transfer(
        [tmp_path / "content"],
        style_path,
        output_dir=tmp_path / "out",
        config=StyleConfig(size=32, steps=3),
        device="cpu",
        workers=2,
        pretrained=False,
    )

    by_name = {Path(r["path"]).name: r for r in records}
    assert set(by_name) == {"broken.jpg", "tall.jpg", "wide.png"}
    assert by_name["broken.jpg"]["status"].startswith("error: UnidentifiedImageError")
    for name in ("tall.jpg", "wide.png"):
        record = by_name[name]
        assert record["status"] == "ok"
        assert Path(record["output_path"]) == tmp_path / "out" / f"{Path(name).stem}_stylized.png"
        assert Path(record["output_path"]).exists()
        assert record["device"] == "cpu"
