import os
import shutil
import random
from PIL import Image

# Step 2a: Preprocessing function
def preprocess_images(input_dir, output_dir, size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for img_file in files:
            if img_file.startswith('.'):
                continue

            img_path = os.path.join(root, img_file)
            rel_path = os.path.relpath(root, input_dir)  # e.g., "Real" or "Fake"
            save_dir = os.path.join(output_dir, rel_path)
            os.makedirs(save_dir, exist_ok=True)

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(size)
                img.save(os.path.join(save_dir, img_file))
                print(f"Processed {img_path} -> {save_dir}")
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

# Step 2b: Split function
def create_train_val_test_split(all_dir, output_dir, split_ratio=(0.7, 0.2, 0.1)):
    classes = os.listdir(all_dir)

    for cls in classes:
        cls_dir = os.path.join(all_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        images = os.listdir(cls_dir)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(split_ratio[0] * n_total)
        n_val = int(split_ratio[1] * n_total)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:]
        }

        for split, files in splits.items():
            split_dir = os.path.join(output_dir, split, cls)
            # Clear old split folder automatically
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)
            os.makedirs(split_dir, exist_ok=True)

            for f in files:
                src = os.path.join(cls_dir, f)
                dst = os.path.join(split_dir, f)
                shutil.copy(src, dst)

    print("✅ Dataset split into train/val/test")

# Step 3: Main execution
if __name__ == "__main__":
    # Preprocess all raw images into "all"
    preprocess_images("data/raw/images", "data/processed/images/all")

    # Split preprocessed images into train/val/test
    create_train_val_test_split("data/processed/images/all", "data/processed/images")

    # scripts/preprocess/split_video_frames.py
from face_align_and_crop import create_train_val_test_split

if __name__ == "__main__":
    create_train_val_test_split("data/processed/frames", "data/processed/video_frames")
