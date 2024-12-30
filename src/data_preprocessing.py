import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(34)

def preprocess_data(df):
    # Drop the 'Time' column and duplicate rows
    df.drop(columns="Time", inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Separate fraud data
    fraud_data = df[df['Class'] == 1]
    
    # Scale features and split into train/test sets
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fraud_data.drop(columns='Class'))
    X = pd.DataFrame(X_scaled, columns=fraud_data.drop(columns='Class').columns)
    y = fraud_data['Class'].reset_index(drop=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=34)
    
    return X_train, X_test, y_train, y_test