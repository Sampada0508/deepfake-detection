import os, shutil

base_path = "data/processed/frames/fake"
min_frames = 10

for sub in os.listdir(base_path):
    sub_path = os.path.join(base_path, sub)
    if os.path.isdir(sub_path):
        frames = [f for f in os.listdir(sub_path) if f.endswith((".jpg", ".png"))]
        if len(frames) < min_frames:
            print(f"🗑 Removing {sub} (only {len(frames)} frames)")
            shutil.rmtree(sub_path)