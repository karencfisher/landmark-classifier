import torch
import torch.nn as nn

def one_batch_train(model, data_loader, device):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    # Get ONE batch
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)

    for step in range(200):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            print(f"step {step:03d} | loss {loss.item():.4f} | acc {acc:.3f}")
