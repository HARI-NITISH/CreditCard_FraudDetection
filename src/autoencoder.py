import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 29),
            nn.BatchNorm1d(29),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.decoder = nn.Sequential(
            nn.Linear(29, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, input_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(29, num_classes), 



            nn.Softmax(dim=1)
        )
        self._initialize_weights()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classification = self.classifier(encoded)
        return decoded, classification

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def fit(self, X_train, epochs=50, batch_size=32, validation_split=0.1):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        X_train = X_train.to_numpy()  # Convert DataFrame to NumPy array
        train_data = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        for epoch in range(epochs):
            self.train()
            for data in train_loader:
                inputs = data[0]
                decoded, _ = self(inputs)
                loss = criterion(decoded, inputs)
                optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.state_dict(), 'best_model.pt')
            else:
                print("Early stopping")
                break

        self.load_state_dict(torch.load('best_model.pt', weights_only=True))