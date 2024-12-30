import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
import numpy as np

def train_model(model, X_train, y_train, X_test, y_test):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, epoch / 10))

    num_epochs = 10000
    best_loss = float('inf')

    train_data = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            inputs, labels = data
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # if loss.item() < best_loss:
        #     best_loss = loss.item()
        #     torch.save(model.state_dict(), 'best_model.pt')
        # else:
        #     print("Early stopping")
        #     break

    model.load_state_dict(torch.load('best_model.pt', weights_only=True))
    return model

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32)
        if isinstance(y_test, np.ndarray):
            labels = torch.tensor(y_test, dtype=torch.long)
        else:
            labels = torch.tensor(y_test.to_numpy(), dtype=torch.long)      
        _, outputs = model(inputs)
        _, y_pred = torch.max(outputs, 1)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, y_pred)
    
    return accuracy*100