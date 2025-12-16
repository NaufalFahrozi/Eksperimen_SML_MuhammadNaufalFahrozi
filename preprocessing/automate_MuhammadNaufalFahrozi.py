"""
Automated Preprocessing Script for Telco Customer Churn Dataset
Author: Muhammad Naufal Fahrozi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def load_data(filepath):
    """Load dataset dari file CSV."""
    df = pd.read_csv(filepath)
    return df


def drop_unnecessary_columns(df):
    """Drop kolom yang tidak diperlukan untuk modelling."""
    df_clean = df.drop('customerID', axis=1)
    return df_clean


def handle_missing_values(df):
    """Handle missing values di kolom TotalCharges."""
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    return df


def encode_categorical(df):
    """Encode kolom categorical ke numerical menggunakan Label Encoding."""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    return df


def scale_numerical_features(df):
    """Standarisasi fitur numerical."""
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df


def split_data(df, test_size=0.2, random_state=42):
    """Split data menjadi training dan testing set."""
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(input_path, output_dir):
    """
    Pipeline lengkap untuk preprocessing data.
    
    Parameters:
    -----------
    input_path : str
        Path ke file CSV raw dataset
    output_dir : str
        Directory untuk menyimpan hasil preprocessing
    
    Returns:
    --------
    tuple : X_train, X_test, y_train, y_test
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = load_data(input_path)
    
    df = drop_unnecessary_columns(df)
    
    df = handle_missing_values(df)
    
    df = encode_categorical(df)
    
    df = scale_numerical_features(df)
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    train_data = X_train.copy()
    train_data['Churn'] = y_train.values
    
    test_data = X_test.copy()
    test_data['Churn'] = y_test.values
    
    train_path = os.path.join(output_dir, 'telco_churn_train.csv')
    test_path = os.path.join(output_dir, 'telco_churn_test.csv')
    full_path = os.path.join(output_dir, 'telco_churn_preprocessed.csv')
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    df.to_csv(full_path, index=False)
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    input_path = os.path.join(parent_dir, 'telco_churn.csv')
    output_dir = current_dir
    
    X_train, X_test, y_train, y_test = preprocess_pipeline(input_path, output_dir)
