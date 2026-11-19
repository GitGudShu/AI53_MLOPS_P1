from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

EXPECTED_SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_dataset(raw_root: Path) -> dict:
    """Validate the raw dataset and return a deterministic quality report.

    Required Core checks:
      1. required splits exist;
      2. exactly the expected classes exist in every split;
      3. expected image counts are present;
      4. every image can be fully decoded by Pillow (`Image.load()` or equivalent);
      5. no split or class is empty.
      """
    root = Path(raw_root)
    classes = sorted(EXPECTED_CLASSES)
    splits_report: dict[str, dict] = {}
    total_images = 0

    for split in sorted(EXPECTED_SPLITS):
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory for '{split}': {split_dir}")

        found_classes = {entry.name for entry in split_dir.iterdir() if entry.is_dir()}
        expected_set = set(EXPECTED_CLASSES)
        if found_classes != expected_set:
            missing = sorted(expected_set - found_classes)
            extra = sorted(found_classes - expected_set)
            raise RuntimeError(
                f"Invalid class directories in split '{split}': "
                f"missing={missing}, extra={extra}"
            )

        class_counts: dict[str, int] = {}
        for cls in classes:
            class_dir = split_dir / cls
            images = sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not images:
                raise RuntimeError(f"Empty class '{cls}' in split '{split}': {class_dir}")

            for img_path in images:
                try:
                    with Image.open(img_path) as img:
                        img.load()
                except Exception as exc:
                    raise OSError(f"Cannot decode image '{img_path}'") from exc

            class_counts[cls] = len(images)

        split_total = sum(class_counts.values())
        expected_total = EXPECTED_SPLITS[split]
        if split_total != expected_total:
            raise RuntimeError(
                f"Split '{split}' has {split_total} images, expected {expected_total}"
            )

        splits_report[split] = {
            "total_images": split_total,
            "class_counts": class_counts,
        }
        total_images += split_total

    return {
        "status": "PASS",
        "classes": classes,
        "splits": splits_report,
        "total_images": total_images,
    }


def main() -> None:
    report = validate_dataset(Path("data/raw"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
