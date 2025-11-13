import os

def check_dataset_structure(base_path):
    print(f"🔍 Scanning dataset in: {base_path}\n")

    for cls in ["real", "fake"]:
        class_path = os.path.join(base_path, cls)
        if not os.path.exists(class_path):
            print(f"❌ Missing folder: {class_path}")
            continue

        subfolders = [f for f in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, f))]
        print(f"📂 Checking {cls} ({len(subfolders)} subfolders) ...")

        empty_count = 0
        min_frames = float("inf")
        max_frames = 0

        for sub in subfolders:
            sub_path = os.path.join(class_path, sub)
            frames = [f for f in os.listdir(sub_path) if f.lower().endswith((".jpg", ".png"))]

            if len(frames) == 0:
                print(f"   ⚠️  Empty folder: {sub_path}")
                empty_count += 1
            else:
                min_frames = min(min_frames, len(frames))
                max_frames = max(max_frames, len(frames))

        print(f"   ➡️  Empty folders: {empty_count}")
        if empty_count < len(subfolders):
            print(f"   ➡️  Frame count range: {min_frames} - {max_frames}\n")


if __name__ == "__main__":
    base_path = "data/processed/frames"
    check_dataset_structure(base_path)