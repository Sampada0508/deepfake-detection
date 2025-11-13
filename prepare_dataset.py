import os
import shutil
from tqdm import tqdm

# Paths to the real and fake frame folders
REAL_FOLDER = "data/processed/frames/real"
FAKE_FOLDER = "data/processed/frames/fake"

# Destination folders for training, validation, and testing
DEST_ROOT = "data/dataset"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_dirs():
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            dir_path = os.path.join(DEST_ROOT, split, label)
            os.makedirs(dir_path, exist_ok=True)

def split_videos(src_folder, label):
    videos = os.listdir(src_folder)
    total = len(videos)
    
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    for idx, video in enumerate(tqdm(videos, desc=f"Processing {label} videos")):
        video_src = os.path.join(src_folder, video)
        
        if idx < train_end:
            split = "train"
        elif idx < val_end:
            split = "val"
        else:
            split = "test"
        
        dest_folder = os.path.join(DEST_ROOT, split, label, video)
        shutil.copytree(video_src, dest_folder)

def main():
    print("Creating dataset folders...")
    create_dirs()
    
    print("Splitting real videos...")
    split_videos(REAL_FOLDER, "real")
    
    print("Splitting fake videos...")
    split_videos(FAKE_FOLDER, "fake")
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()