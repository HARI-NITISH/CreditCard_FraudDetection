import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set seed for reproducibility
torch.manual_seed(34)
np.random.seed(34)
class Generator(nn.Module):
    def __init__(self, latent_dim, out_shape, num_classes):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128, momentum=0.8),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.Linear(512, out_shape),
            nn.Tanh(),
        )
    def forward(self, noise, labels):
        label_input = self.label_embedding(labels)
        gen_input = noise * label_input
        return self.model(gen_input)
    
class Discriminator(nn.Module):
    def __init__(self, out_shape, num_classes):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, out_shape)

        self.model = nn.Sequential(
            nn.Linear(out_shape, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, features, labels):
        label_input = self.label_embedding(labels)
        d_input = features * label_input
        return self.model(d_input)
class cGAN:
    def __init__(self, out_shape, num_classes=2, latent_dim=32):
        self.latent_dim = latent_dim
        self.out_shape = out_shape
        self.num_classes = num_classes

        # Initialize generator and discriminator
        self.generator = Generator(latent_dim, out_shape, num_classes)
        self.discriminator = Discriminator(out_shape, num_classes)

        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Loss function
        self.adversarial_loss = nn.BCELoss()

    def train(self, X_train, y_train, pos_index, neg_index, epochs=200, batch_size=32, sample_interval=200):
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)

        # Create DataLoader
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for i, (samples, labels) in enumerate(dataloader):
                # Adversarial ground truths
                valid = torch.ones((samples.size(0), 1), dtype=torch.float32)
                fake = torch.zeros((samples.size(0), 1), dtype=torch.float32)

                # Train Discriminator
                self.optimizer_D.zero_grad()

                # Real samples
                real_loss = self.adversarial_loss(self.discriminator(samples, labels), valid)

                # Generate fake samples
                noise = torch.randn((samples.size(0), self.latent_dim))
                fake_labels = torch.randint(0, self.num_classes, (samples.size(0),))
                gen_samples = self.generator(noise, fake_labels)

                # Fake samples
                fake_loss = self.adversarial_loss(self.discriminator(gen_samples.detach(), fake_labels), fake)

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                # Train Generator
                self.optimizer_G.zero_grad()

                # Generate fake samples for generator training
                gen_validity = self.discriminator(gen_samples, fake_labels)
                g_loss = self.adversarial_loss(gen_validity, valid)
                g_loss.backward()
                self.optimizer_G.step()

            # Print progress
            if (epoch + 1) % sample_interval == 0:
                print(f"[Epoch {epoch + 1}/{epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    def generate_synthetic_data(self, n_samples):
        noise = torch.randn((n_samples, self.latent_dim))
        sampled_labels = torch.randint(0, self.num_classes, (n_samples,))
        synthetic_data = self.generator(noise, sampled_labels)
        synthetic_data = synthetic_data.detach().numpy()
        return synthetic_data, sampled_labels.numpy()

    def test(self, X_test, y_test):
            # Convert to PyTorch tensors
            X_test = torch.tensor(X_test.values, dtype=torch.float32)
            
            if isinstance(y_test, np.ndarray):
                y_test = torch.tensor(y_test, dtype=torch.long)
            else:
                y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)

            # Create DataLoader
            dataset = TensorDataset(X_test, y_test)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

            # Initialize lists to store predictions and labels
            all_preds = []
            all_labels = []

            # Evaluate the model
            self.discriminator.eval()
            with torch.no_grad():
                for samples, labels in dataloader:
                    preds = self.discriminator(samples, labels)
                    preds = (preds > 0.5).float()
                    all_preds.extend(preds.numpy())
                    all_labels.extend(labels.numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            # Calculate accuracy
            accuracy = accuracy_score(all_labels, all_preds)
            
            return accuracy
def generate_synthetic_data(X, y):
    data = pd.concat([X, y], axis=1)
    out_shape = X.shape[1]
    cgan = cGAN(out_shape)
    cgan.train(X, y, pos_index=y[y == 1].index, neg_index=y[y == 0].index, epochs=1000, batch_size=32, sample_interval=200)

    # Generate synthetic data
    synthetic_data, synthetic_labels = cgan.generate_synthetic_data(int(0.2*len(data)))

    # Convert synthetic data to DataFrame
    synthetic_data = pd.DataFrame(synthetic_data, columns=X.columns)
    synthetic_data['Class'] = pd.Series(synthetic_labels)

    # Combine original and synthetic data
    combined_data = pd.concat([data, synthetic_data], ignore_index=True)

    # Split into train and test sets
    train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)

    X_train = train_data.drop('Class', axis=1)
    y_train = train_data['Class']
    X_test = test_data.drop('Class', axis=1)
    y_test = test_data['Class']

    return X_train, X_test, y_train, y_test