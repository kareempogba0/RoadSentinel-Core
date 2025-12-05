import os
from pathlib import Path


def analyze_split(split_name, split_path):
    labels_path = Path(split_path) / "labels"

    total_images = 0
    total_objects = 0
    empty_images = 0  # Background images

    if not labels_path.exists():
        print(f"Skipping {split_name} (labels folder not found)")
        return

    label_files = list(labels_path.glob("*.txt"))
    total_images = len(label_files)

    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            # Filter out empty lines just in case
            lines = [l.strip() for l in lines if l.strip()]

            if len(lines) == 0:
                empty_images += 1
            else:
                total_objects += len(lines)

    print(f"{'=' * 40}")
    print(f"📊 STATISTICS FOR: {split_name.upper()}")
    print(f"{'=' * 40}")
    print(f"Total Images:      {total_images}")
    print(f"Images w/ Objects: {total_images - empty_images}")
    print(f"Background Images: {empty_images} (No accidents)")
    print(f"Total Labels (Boxes): {total_objects}")
    print(f"Avg Objects/Image: {total_objects / total_images:.2f}" if total_images > 0 else "0")
    print("\n")


def main():
    # Define root path
    root_dir = Path("data/car-accident-detection-1")

    splits = ["train", "valid", "test"]

    print("\nStarting Dataset Analysis...\n")

    for split in splits:
        analyze_split(split, root_dir / split)


if __name__ == "__main__":
    main()