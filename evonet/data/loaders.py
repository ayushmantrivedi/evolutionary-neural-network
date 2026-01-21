"""
Dataset Loaders Module

Functions for loading various datasets (sklearn built-in and custom).
"""

import numpy as np
import pandas as pd
import json
import sys
import os
import logging
from typing import Tuple, List, Optional, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import (
    load_breast_cancer, 
    fetch_california_housing, 
    load_iris, 
    load_wine, 
    load_digits
)
from sklearn.preprocessing import StandardScaler

from evonet.config import get_output_directory
from evonet.data.preprocessing import preprocess_dataset, clean_dataset

# Configure logging
logger = logging.getLogger(__name__)

# Global variable to track target scaling
target_scaler_global = None


def load_housing_dataset(
    custom_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess the California Housing Dataset.
    
    Args:
        custom_path: Optional path to custom housing CSV
        
    Returns:
        Tuple of (X, y) arrays
    """
    if custom_path is not None:
        print(f"\nLoading custom dataset from: {custom_path}")
        try:
            df = pd.read_csv(custom_path)
        except FileNotFoundError:
            print(f"Error: File not found at {custom_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading CSV file {custom_path}: {e}")
            sys.exit(1)
        
        df = clean_dataset(df)
        feature_df = df.iloc[:, :-1]
        feature_df = pd.get_dummies(feature_df)
        
        scaler = StandardScaler()
        X = scaler.fit_transform(feature_df.values.astype(np.float64))
        y = df.iloc[:, -1].values.reshape(-1, 1).astype(np.float64)
        
        print(f"Number of samples: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        
        X, y, feature_names = preprocess_dataset(X, y, "Custom Housing")
        return X, y
    else:
        housing = fetch_california_housing()
        X = housing.data
        y = housing.target.reshape(-1, 1)
        
        print("\nDataset Information:")
        print(f"Number of samples: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        print("\nFeature names:")
        for i, name in enumerate(housing.feature_names):
            print(f"{i+1}. {name}")
        print("\nTarget: Housing Value")
        
        X, y, feature_names = preprocess_dataset(X, y, "California Housing")
        return X, y


def load_cancer_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess the Breast Cancer Dataset.
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    print("\nLoading breast cancer dataset (sklearn)...")
    data = load_breast_cancer()
    X = data.data
    y = data.target.reshape(-1, 1)
    feature_names = list(data.feature_names)
    
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print("\nTarget: Cancer (0=malignant, 1=benign)")
    
    X, y, feature_names = preprocess_dataset(X, y, "Breast Cancer")
    return X, y, feature_names


def load_iris_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess the Iris Dataset (multi-class classification).
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    print("\nLoading iris dataset (sklearn)...")
    data = load_iris()
    X = data.data
    y = data.target.reshape(-1, 1)
    feature_names = list(data.feature_names)
    target_names = data.target_names
    
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print("\nTarget classes:")
    for i, name in enumerate(target_names):
        print(f"  Class {i}: {name}")
    
    # Create visualization
    output_dir = get_output_directory()
    unique, counts = np.unique(y, return_counts=True)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar([target_names[i] for i in unique], counts, color=['red', 'green', 'blue'])
    plt.xlabel('Iris Species')
    plt.ylabel('Count')
    plt.title('Iris Dataset Class Distribution')
    
    plt.subplot(1, 2, 2)
    for i, target_name in enumerate(target_names):
        class_data = X[y.ravel() == i]
        plt.scatter(class_data[:, 0], class_data[:, 1], label=target_name, alpha=0.7)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Iris Dataset: Sepal Length vs Sepal Width')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iris_dataset_analysis.png'), bbox_inches='tight')
    plt.close()
    
    print(f"Iris dataset analysis saved to: iris_dataset_analysis.png")
    
    X, y, feature_names = preprocess_dataset(X, y, "Iris")
    return X, y, feature_names


def load_wine_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess the Wine Dataset (multi-class classification).
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    import seaborn as sns
    
    print("\nLoading wine dataset (sklearn)...")
    data = load_wine()
    X = data.data
    y = data.target.reshape(-1, 1)
    feature_names = list(data.feature_names)
    target_names = data.target_names
    
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    
    # Create visualization
    output_dir = get_output_directory()
    unique, counts = np.unique(y, return_counts=True)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.bar([target_names[i] for i in unique], counts, color=['red', 'green', 'blue'])
    plt.xlabel('Wine Type')
    plt.ylabel('Count')
    plt.title('Wine Dataset Class Distribution')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    for i, target_name in enumerate(target_names):
        class_data = X[y.ravel() == i]
        plt.scatter(class_data[:, 0], class_data[:, 1], label=target_name, alpha=0.7)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Wine Dataset: Alcohol vs Malic Acid')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    correlation_matrix = np.corrcoef(X.T)
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm',
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wine_dataset_analysis.png'), bbox_inches='tight')
    plt.close()
    
    X, y, feature_names = preprocess_dataset(X, y, "Wine")
    return X, y, feature_names


def load_digits_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess the Digits Dataset (multi-class classification).
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    print("\nLoading digits dataset (sklearn)...")
    data = load_digits()
    X = data.data
    y = data.target.reshape(-1, 1)
    feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
    
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    output_dir = get_output_directory()
    unique, counts = np.unique(y, return_counts=True)
    
    plt.figure(figsize=(8, 6))
    plt.bar(unique.ravel(), counts, color='skyblue')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.title('Digits Dataset Class Distribution')
    plt.xticks(unique.ravel())
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'digits_dataset_analysis.png'), bbox_inches='tight')
    plt.close()
    
    X, y, feature_names = preprocess_dataset(X, y, "Digits")
    return X, y, feature_names


def load_custom_csv_dataset(custom_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess a custom CSV dataset.
    
    Args:
        custom_path: Path to CSV file
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    global target_scaler_global
    
    print(f"\nLoading custom dataset from: {custom_path}")
    
    try:
        df = pd.read_csv(custom_path, encoding='latin1', nrows=2000)
    except FileNotFoundError:
        print(f"Error: File not found at {custom_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file {custom_path}: {e}")
        sys.exit(1)
    
    if df.empty:
        print("Error: No data found in the CSV file.")
        sys.exit(1)
    
    df = clean_dataset(df)
    
    if df.empty:
        print("Error: No data remaining after cleaning.")
        sys.exit(1)
    
    # Exclude ID columns
    cols_to_exclude = [col for col in df.columns if 'id' in col.lower()]
    print(f"Excluding columns from features (ID-like): {cols_to_exclude}")
    
    feature_df = df.iloc[:, :-1].copy()
    feature_df = feature_df.drop(columns=cols_to_exclude, errors='ignore')
    
    if feature_df.empty:
        print("Error: No features remaining after removing ID columns.")
        sys.exit(1)
    
    # Process features
    try:
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
        
        numeric_df = feature_df[numeric_cols] if numeric_cols else pd.DataFrame()
        
        if categorical_cols:
            categorical_df = feature_df[categorical_cols].dropna()
            if not categorical_df.empty:
                dummies_df = pd.get_dummies(categorical_df, drop_first=False)
            else:
                dummies_df = pd.DataFrame()
        else:
            dummies_df = pd.DataFrame()
        
        if not numeric_df.empty and not dummies_df.empty:
            feature_df = pd.concat([numeric_df, dummies_df], axis=1)
        elif not numeric_df.empty:
            feature_df = numeric_df
        elif not dummies_df.empty:
            feature_df = dummies_df
        else:
            print("Error: No valid features found after processing.")
            sys.exit(1)
    except Exception as e:
        print(f"Error processing features: {e}")
        feature_df = feature_df.select_dtypes(include=[np.number])
        if feature_df.empty:
            sys.exit(1)
    
    # Sample if too large
    if len(feature_df) > 2000:
        print(f"⚠️  Large dataset detected. Sampling 2000 random samples.")
        sampled = feature_df.sample(n=2000, random_state=42)
        feature_df = sampled
        df = df.loc[sampled.index]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df.values.astype(np.float64))
    
    target_col = df.iloc[:, -1]
    
    if target_col.dtype == 'object':
        print(f"Target column is categorical. Encoding...")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(target_col.values).reshape(-1, 1)
    else:
        y_raw = target_col.values.reshape(-1, 1)
        y_max = np.max(np.abs(y_raw))
        
        if y_max > 1e6:
            print(f"⚠️  Large target values detected. Scaling for stability.")
            target_scaler = StandardScaler()
            y = target_scaler.fit_transform(y_raw)
            target_scaler_global = target_scaler
        else:
            y = y_raw
            target_scaler_global = None
    
    feature_names = feature_df.columns.tolist()
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    
    X, y, feature_names = preprocess_dataset(X, y, "Custom CSV")
    return X, y, feature_names


def load_telemetry_dataset(json_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess a telemetry JSON dataset.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    print(f"\nLoading telemetry dataset from: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(data)} telemetry records")
    
    # Convert to DataFrame
    records = []
    for record in data:
        record_dict = {
            'deviceID': record['deviceID'],
            'deviceType': record['deviceType'],
            'timestamp': record['timestamp'],
            'country': record['location']['country'],
            'city': record['location']['city'],
            'area': record['location']['area'],
            'factory': record['location']['factory'],
            'section': record['location']['section'],
            'temperature': record['data']['temperature'],
            'status': record['data']['status']
        }
        records.append(record_dict)
    
    df = pd.DataFrame(records)
    
    # Extract time features
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    # Target: healthy = 1, unhealthy = 0
    df['target'] = (df['status'] == 'healthy').astype(int)
    
    df = clean_dataset(df)
    
    feature_cols = ['deviceType', 'country', 'city', 'area', 'factory', 'section',
                   'temperature', 'hour', 'day_of_week', 'month']
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    feature_df = df[feature_cols].copy()
    
    # Process features
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
    
    numeric_df = feature_df[numeric_cols] if numeric_cols else pd.DataFrame()
    
    if categorical_cols:
        dummies_df = pd.get_dummies(feature_df[categorical_cols], drop_first=False)
    else:
        dummies_df = pd.DataFrame()
    
    if not numeric_df.empty and not dummies_df.empty:
        feature_df = pd.concat([numeric_df, dummies_df], axis=1)
    elif not numeric_df.empty:
        feature_df = numeric_df
    else:
        feature_df = dummies_df
    
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df.values.astype(np.float64))
    y = df['target'].values.reshape(-1, 1)
    
    feature_names = feature_df.columns.tolist()
    
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    
    X, y, feature_names = preprocess_dataset(X, y, "Telemetry")
    return X, y, feature_names
