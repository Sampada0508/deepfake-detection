import torch
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import os, csv
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paths
DATA_DIR = "data/dataset/test"
MODEL_PATH = "models/deepfake_resnet18.pth"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_OUT = os.path.join(OUT_DIR, "video_predictions.csv")

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Transform
tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

# Dataset & loader (uses folder structure: test/real, test/fake)
dataset = datasets.ImageFolder(DATA_DIR, transform=tf)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

# Load model (ResNet18 with 2-class head)
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

idx_to_class = {v:k for k,v in dataset.class_to_idx.items()}

all_preds = []
all_labels = []
filenames = []

with torch.no_grad():
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    # filenames are in dataset.imgs in the same order as loader's output order
for path, _ in dataset.imgs:
    filenames.append(os.path.basename(path))

# Save CSV: filename, true_label, predicted_label
with open(CSV_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename","true","pred"])
    for fname, true_idx, pred_idx in zip(filenames, all_labels, all_preds):
        writer.writerow([fname, idx_to_class[true_idx], idx_to_class[pred_idx]])

# Print evaluation
acc = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {acc*100:.2f}%")
print("\nClassification report:\n")
print(classification_report(all_labels, all_preds, target_names=[idx_to_class[i] for i in sorted(idx_to_class)]))
print("\nConfusion matrix:\n")
print(confusion_matrix(all_labels, all_preds))
print(f"\nSaved predictions to {CSV_OUT}")
