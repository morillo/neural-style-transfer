"""Stylise the bundled sample photos with Van Gogh's Starry Night.

pip install -e .
python examples/basic_usage.py
"""

from pathlib import Path

from neural_style_transfer import StyleConfig, run_style_transfer

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def main() -> None:
    records = run_style_transfer(
        content=[SAMPLES / "content"],
        style_image=SAMPLES / "style" / "starry_night.jpg",
        output_dir="outputs",
        config=StyleConfig(size=512, steps=300),
    )
    for record in records:
        print(f"{record['status']:>4}  {record['seconds']:5.1f}s  {record['device']}  {record['output_path']}")


if __name__ == "__main__":
    main()
