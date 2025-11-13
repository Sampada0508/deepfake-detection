# scripts/preprocess/split_video_frames.py
from face_align_and_crop import create_train_val_test_split
import os

if __name__ == "__main__":
    create_train_val_test_split(
        os.path.join("data", "processed", "frames", "Real_flat"),  # flattened Real folder
        os.path.join("data", "processed", "video_frames", "train", "Real")  # destination
    )

    create_train_val_test_split(
        os.path.join("data", "processed", "frames", "Fake_flat"),  # flattened Fake folder
        os.path.join("data", "processed", "video_frames", "train", "Fake")  # destination
    )