import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Paths
test_dir = "data/processed/images/test"

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load test dataset
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load trained model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes: real/fake
model.load_state_dict(torch.load("models/image/image_model.pth"))
model.eval()

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# Initialize counters
class_correct = [0, 0]
class_total = [0, 0]

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(labels)):
            label = labels[i].item()
            class_correct[label] += (predicted[i].item() == label)
            class_total[label] += 1

# Print per-class accuracy
for i, class_name in enumerate(test_dataset.classes):
    accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"{class_name} Accuracy: {accuracy:.2f}%")
