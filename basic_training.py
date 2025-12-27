import torch

def train(model, device, optimizer, data_loader, criterion):
    model.to(device)
    num_epochs = 5

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()  # always start epoch in training mode

        for images, labels in data_loader:  # your DataLoader
            images = images.to(device)
            labels = labels.to(device)
        
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        print(f"Epoch {epoch+1}/{num_epochs} — Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")