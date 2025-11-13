import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Paths
data_dir = "data/processed/images/test"  # Change if needed
model_path = "models/image/image_model.pth"

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset and DataLoader
test_dataset = datasets.ImageFolder(data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes: real, fake
model.load_state_dict(torch.load(model_path))
model.eval()

# Move model to device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Inference
results = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        results.extend(preds.cpu().numpy())

# Map indices to class names
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
pred_labels = [idx_to_class[i] for i in results]

# Print results
for img_path, pred in zip(test_dataset.imgs, pred_labels):
    print(f"{os.path.basename(img_path[0])}: {pred}")


    import csv

with open("image_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "prediction"])
    for img_path, pred in zip(test_dataset.imgs, pred_labels):
        writer.writerow([os.path.basename(img_path[0]), pred])

print("✅ Predictions saved to image_predictions.csv")