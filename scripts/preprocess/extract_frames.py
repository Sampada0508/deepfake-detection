import os
import cv2
import shutil
import random

def extract_frames(video_dir, output_dir, frame_rate=1):
    """
    Extract frames from videos.
    frame_rate=1 means 1 frame per second.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for cls in os.listdir(video_dir):
        cls_dir = os.path.join(video_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        for video_file in os.listdir(cls_dir):
            if video_file.startswith('.'):
                continue
            video_path = os.path.join(cls_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open video {video_path}")
                continue

            count = 0
            saved_count = 0
            video_name = os.path.splitext(video_file)[0]
            save_dir = os.path.join(output_dir, cls, video_name)
            os.makedirs(save_dir, exist_ok=True)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if count % frame_rate == 0:
                    frame_path = os.path.join(save_dir, f"{count:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    saved_count += 1
                count += 1

            cap.release()
            print(f"Extracted {saved_count} frames from {video_file} -> {save_dir}")

# Optional: split frames into train/val/test later

if __name__ == "__main__":
    extract_frames("data/raw/videos", "data/processed/frames", frame_rate=30)
    

