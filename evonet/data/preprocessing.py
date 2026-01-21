"""
Data Preprocessing Module

Functions for cleaning, preprocessing, and preparing datasets for training.
Includes SMOTE balancing for imbalanced classification datasets.
"""

import numpy as np
import logging
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)

# SMOTE availability check
SMOTE_AVAILABLE = False
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    logger.info("SMOTE successfully imported and available!")
except ImportError as e:
    logger.warning(f"SMOTE not available: {e}")
except Exception as e:
    logger.warning(f"SMOTE import failed: {e}")


def detect_problem_type(y: np.ndarray) -> str:
    """
    Detect the problem type from target values.
    
    Args:
        y: Target array
        
    Returns:
        str: One of 'binary_classification', 'multi_class_classification', or 'regression'
    """
    unique_vals = np.unique(y)
    
    if len(unique_vals) == 2:
        return 'binary_classification'
    elif len(unique_vals) <= 10:
        return 'multi_class_classification'
    else:
        return 'regression'


def preprocess_dataset(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "Dataset"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Centralized preprocessing function for all datasets.
    
    Handles cleaning, scaling, and SMOTE balancing if needed.
    
    Args:
        X: Features
        y: Target variable
        dataset_name: Name of the dataset for logging
        
    Returns:
        Tuple of (X_processed, y_processed, feature_names)
    """
    print(f"\n=== {dataset_name} Preprocessing ===")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y).flatten()
    
    # Detect problem type
    unique_values = len(np.unique(y))
    is_regression = unique_values > 10
    is_multi_class = unique_values > 2 and unique_values <= 10
    
    if is_regression:
        print("Regression dataset detected. Skipping SMOTE (not applicable).")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Preprocessing complete. Final shape: {X_scaled.shape}")
        return X_scaled, y, [f"Feature_{i}" for i in range(X_scaled.shape[1])]
    
    elif is_multi_class:
        print("Multi-class classification detected. Skipping SMOTE to avoid confusion.")
        print("Using original data distribution for better model learning.")
        
        unique, counts = np.unique(y, return_counts=True)
        print(f"Original class distribution: {dict(zip(unique, counts))}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"Preprocessing complete. Final shape: {X_scaled.shape}")
        return X_scaled, y, [f"Feature_{i}" for i in range(X_scaled.shape[1])]
    
    else:
        # Binary classification
        unique, counts = np.unique(y, return_counts=True)
        print(f"Original class distribution: {dict(zip(unique, counts))}")
        
        min_count = np.min(counts)
        max_count = np.max(counts)
        imbalance_ratio = min_count / max_count
        print(f"Imbalance ratio: {imbalance_ratio:.3f}")
        
        # Apply SMOTE if imbalanced
        if imbalance_ratio < 0.3:
            print("⚠️  WARNING: Binary classification dataset is imbalanced!")
            if SMOTE_AVAILABLE:
                print("Applying SMOTE to balance the dataset...")
                try:
                    smote = SMOTE(random_state=42)
                    X_balanced, y_balanced = smote.fit_resample(X, y)
                    
                    unique_after, counts_after = np.unique(y_balanced, return_counts=True)
                    print(f"After SMOTE - Class distribution: {dict(zip(unique_after, counts_after))}")
                    print("✅ SMOTE applied successfully!")
                    
                    X, y = X_balanced, y_balanced
                except Exception as e:
                    print(f"❌ Error applying SMOTE: {e}")
                    print("⚠️  Continuing with original imbalanced dataset...")
            else:
                print("❌ SMOTE not available. Cannot balance imbalanced dataset.")
                print("⚠️  WARNING: Training on imbalanced dataset may lead to poor performance!")
                print("💡 Suggestion: Install imbalanced-learn package: pip install imbalanced-learn")
        else:
            print("✅ Binary classification dataset is balanced. No SMOTE needed.")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"Preprocessing complete. Final shape: {X_scaled.shape}")
        return X_scaled, y, [f"Feature_{i}" for i in range(X_scaled.shape[1])]


def clean_dataset(df) -> 'pd.DataFrame':
    """
    Clean a pandas DataFrame.
    
    Operations:
    - Remove duplicate rows
    - Drop columns with >50% missing values
    - Remove constant columns
    - Impute missing values (mean for numeric, mode for categorical)
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    import pandas as pd
    
    print("\n--- Cleaning Dataset ---")
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows.")
    
    # Drop columns with >50% missing
    thresh = int(0.5 * len(df))
    before_cols = df.shape[1]
    df = df.dropna(axis=1, thresh=thresh)
    print(f"Dropped {before_cols - df.shape[1]} columns with >50% missing values.")
    
    # Remove constant columns
    nunique = df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=const_cols)
    if const_cols:
        print(f"Removed constant columns: {const_cols}")
    
    # Impute missing values
    for col in df.columns:
        if df[col].dtype == 'O':
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)
        else:
            mean = df[col].mean()
            df[col] = df[col].fillna(mean)
    
    print("Imputed missing values (mean for numeric, mode for categorical).")
    
    return df
