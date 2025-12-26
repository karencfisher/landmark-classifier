import torch
from torch import nn
from tqdm import tqdm


def train_model(model, device, data_loaders, learning_rate, epochs):
    '''
    Train the model
    
    :param model: the model to be trained
    :param data_loader: dictionary of data loaders (train and validation)
    :param learning_rate: learning rate (float)
    :param epochs: number of epochs (int)
    
    Returns train_losses, valid_losses
    '''
    print(f'Learning rate: {learning_rate}')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = 0
        model.train()
        for inputs, labels in tqdm(data_loaders['train'], desc='Training', ascii=' ='):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        valid_loss, accuracy = validate_model(model, device, data_loaders['validate'])

        train_losses.append(train_loss/len(data_loaders['train']))
        valid_losses.append(valid_loss)
        print(f'Train loss: {train_losses[-1]:.3f} - Valid loss: {valid_losses[-1]:.3f} - ',
              f'Valid accuracy: {accuracy * 100:.2f}%\n')
    return train_losses, valid_losses

def validate_model(model, device, data_loader):
    '''
    Test the model, eithr on test or validation
    
    :param model: the model to test
    :param data_loader: the valid dataloader
    
    Returns loss, accuracy
    '''
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Validating', ascii=' ='):
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            test_loss += loss.item()

            ps = torch.exp(logps)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    loss = test_loss/len(data_loader)
    accuracy = test_accuracy/len(data_loader)
    return loss, accuracy