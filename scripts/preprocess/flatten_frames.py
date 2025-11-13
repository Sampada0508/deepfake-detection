# scripts/preprocess/flatten_frames.py
import os
import shutil

source_dir = "data/processed/frames"
for cls in ["Real", "Fake"]:
    cls_dir = os.path.join(source_dir, cls)
    flat_dir = os.path.join(source_dir, cls + "_flat")
    os.makedirs(flat_dir, exist_ok=True)

    for video_folder in os.listdir(cls_dir):
        video_path = os.path.join(cls_dir, video_folder)
        if not os.path.isdir(video_path):
            continue
        for img_file in os.listdir(video_path):
            if img_file.endswith(".jpg"):
                src = os.path.join(video_path, img_file)
                dst = os.path.join(flat_dir, img_file)
                shutil.copy(src, dst)

print("✅ Frames flattened into Real_flat / Fake_flat")