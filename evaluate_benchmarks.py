
import sys
import os
import numpy as np
import time
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Ensure we can import evonet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evonet.core.network import MultiClassEvoNet, RegressionEvoNet
from evonet.core.gpu_backend import is_gpu_available, get_device

def run_benchmark(name, load_func, problem_type='classification', epochs=30):
    print(f"\n{'='*50}")
    print(f"Running Benchmark: {name}")
    print(f"{'='*50}")
    
    # Load data
    data = load_func()
    X, y = data.data, data.target
    
    # Preprocess
    if problem_type == 'classification':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # One-hot encode for training
        encoder = OneHotEncoder(sparse_output=False)
        y_oh = encoder.fit_transform(y.reshape(-1, 1))
        
        X_train, X_test, y_train, y_test, y_oh_train, y_oh_test = train_test_split(
            X, y, y_oh, test_size=0.2, random_state=42
        )
        
        num_classes = y_oh.shape[1]
        print(f"Classes: {num_classes}, Features: {X.shape[1]}")
        
        # Initialize Network
        start_time = time.time()
        print(f"Device: {get_device()}")
        net = MultiClassEvoNet(input_dim=X.shape[1], num_classes=num_classes)
        
        # Train
        print(f"Training for {epochs} epochs...")
        net.train(X_train, y_train, y_oh_train, epochs=epochs, X_val=X_test, y_val=y_test, y_val_oh=y_oh_test)
        
        # Final Evaluation
        acc, loss = net.evaluate(X_test, y_test, y_oh_test)
        end_time = time.time()
        
        print(f"\nFinal Test Accuracy: {acc*100:.2f}%")
        print(f"Time Taken: {end_time - start_time:.2f} seconds")
        
        return acc
        
    elif problem_type == 'regression':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Scale target for better regression convergence
        y_mean = np.mean(y)
        y_std = np.std(y)
        y = (y - y_mean) / y_std
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Features: {X.shape[1]}")
        
        # Initialize Network
        start_time = time.time()
        print(f"Device: {get_device()}")
        net = RegressionEvoNet(input_dim=X.shape[1])
        
        # Train
        print(f"Training for {epochs} epochs...")
        net.train(X_train, y_train, epochs=epochs, X_val=X_test, y_val=y_test)
        
        # Final Evaluation
        loss, _ = net.evaluate(X_test, y_test)
        
        # Convert MSE back to original scale (approximate) just for sanity check context (rmse)
        rmse_original = np.sqrt(loss) * y_std
        
        end_time = time.time()
        
        print(f"\nFinal Test MSE (scaled): {loss:.4f}")
        print(f"Approx RMSE (original units): {rmse_original:.4f}")
        print(f"Time Taken: {end_time - start_time:.2f} seconds")
        
        return loss

if __name__ == "__main__":
    if is_gpu_available():
        print("🚀 GPU Detected! Benchmarking with CUDA acceleration.")
    else:
        print("⚠️ GPU Not Detected. Running on CPU (this might be slower).")

    # 1. Iris (Simple Multi-class)
    run_benchmark("Iris Dataset", load_iris, 'classification', epochs=5)

    # 2. Breast Cancer (Binary Classification)
    # run_benchmark("Breast Cancer Wisconsin", load_breast_cancer, 'classification', epochs=5)

    # 3. Digits (Larger Multi-class - good for GPU)
    # run_benchmark("Digits (8x8 Images)", load_digits, 'classification', epochs=5)

    # 4. California Housing (Regression)
    # run_benchmark("California Housing", fetch_california_housing, 'regression', epochs=5)
