import os
import sys
import pandas as pd
import torch
from src.autoencoder import Autoencoder
from src.model import train_model, evaluate
from src.synthetic_data_gan import generate_synthetic_data, cGAN
from src.data_preprocessing import preprocess_data

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    # Load and preprocess data
    data = pd.read_csv('data/creditcard.csv')
    print(data.head())
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Determine input dimension
    input_dim = X_train.shape[1]
    
    # Generate synthetic data
    c_X_train, c_X_test, c_y_train, c_y_test = generate_synthetic_data(X_train, y_train)
    
    # Test the synthetic data GAN model
    cgan = cGAN(input_dim)
    test_accuracy = cgan.test(X_test, y_test)
    print(f"GAN Test accuracy: {test_accuracy}")
    
    # Initialize and train the autoencoder
    autoencoder = Autoencoder(input_dim=input_dim, num_classes=2)
    autoencoder.fit(c_X_train, epochs=50, batch_size=32, validation_split=0.1)
    
    # Encode the data
    autoencoder.eval()
    with torch.no_grad():
        encoded_X_train = autoencoder.encoder(torch.tensor(c_X_train.to_numpy(), dtype=torch.float32)).numpy()
        encoded_X_test = autoencoder.encoder(torch.tensor(c_X_test.to_numpy(), dtype=torch.float32)).numpy()
    
    # Convert Series to NumPy array before converting to tensor
    c_y_train = c_y_train.to_numpy()
    c_y_test = c_y_test.to_numpy()
    
    # Train the model using the encoded data
    model = train_model(autoencoder, encoded_X_train, c_y_train, encoded_X_test, c_y_test)
    
    # Evaluate the model
    evaluation = evaluate(model, encoded_X_test, c_y_test)
    print(f"Model evaluation accuracy: {evaluation}")

if __name__ == "__main__":
    main()