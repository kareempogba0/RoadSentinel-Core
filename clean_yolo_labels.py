import os

LABELS_DIR = r"E:\RoadSentinel-Core\data\car-accident-detection-1\train\labels"

def is_valid_yolo_line(parts):
    """Check if a line has valid YOLO format."""
    if len(parts) != 5:
        return False
    try:
        int(parts[0])  # class id
        float(parts[1])
        float(parts[2])
        float(parts[3])
        float(parts[4])
        return True
    except:
        return False

total_fixed = 0

for filename in os.listdir(LABELS_DIR):
    if not filename.endswith(".txt"):
        continue

    path = os.path.join(LABELS_DIR, filename)

    with open(path, "r") as f:
        lines = f.readlines()

    cleaned = []
    for line in lines:
        line = line.strip()  # remove spaces + blank lines
        if not line:
            continue

        parts = line.split()

        # Fix if extra spaces exist
        parts = [p.strip() for p in parts]

        # Validate YOLO format
        if not is_valid_yolo_line(parts):
            print(f"[SKIPPED] Invalid line in {filename}: {line}")
            continue

        # Force single class = 0
        parts[0] = "0"

        cleaned.append(" ".join(parts))

    # Write cleaned version
    with open(path, "w") as f:
        f.write("\n".join(cleaned))

    total_fixed += 1
    print(f"[CLEANED] {filename}")

print(f"\nDone! Cleaned {total_fixed} annotation files.")
