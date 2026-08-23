import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, _, num_classes = get_dataloaders(batch_size=64, size=128)
model = build_model(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device) # Reshape labels for loss function
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

# Save the trained weights
torch.save(model.state_dict(), "pathmnist_resnet18.pth")
print("Model saved to pathmnist_resnet18.pth")