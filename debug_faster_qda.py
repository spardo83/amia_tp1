#!/usr/bin/env python3
"""
Debug específico para FasterQDA
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Importar implementaciones finales
from implementations_final import *

def get_wine_dataset():
    data = load_wine()
    X_full = data.data
    y_full = np.array([data.target_names[y] for y in data.target.reshape(-1,1)])
    return X_full, y_full

def label_encode(y_full):
    return LabelEncoder().fit_transform(y_full.flatten()).reshape(y_full.shape)

def split_transpose(X, y, test_size, random_state):
    return [elem.T for elem in train_test_split(X, y, test_size=test_size, random_state=random_state)]

def debug_faster_qda():
    """Debug específico para FasterQDA"""
    print("=== Debug FasterQDA ===")
    
    # Cargar datos
    X_full, y_full = get_wine_dataset()
    y_full_encoded = label_encode(y_full)
    X_train, X_test, y_train, y_test = split_transpose(
        X_full, y_full_encoded, test_size=0.3, random_state=42
    )
    
    # Entrenar modelo
    model = FasterQDA()
    model.fit(X_train, y_train)
    
    print(f"X_test shape: {X_test.shape}")
    print(f"tensor_means shape: {model.tensor_means.shape}")
    print(f"tensor_inv_cov shape: {model.tensor_inv_cov.shape}")
    
    # Probar con una observación específica
    class_idx = 0
    obs_idx = 0
    
    print(f"\nProbando clase {class_idx}, observación {obs_idx}")
    
    # Calcular unbiased_x para esta clase
    unbiased_x = X_test - model.tensor_means[class_idx:class_idx+1, :, :]
    print(f"unbiased_x shape: {unbiased_x.shape}")
    
    # Tomar una observación específica
    x_i = unbiased_x[:, obs_idx:obs_idx+1]
    print(f"x_i shape: {x_i.shape}")
    
    # Obtener la matriz de covarianza
    inv_cov = model.tensor_inv_cov[class_idx]
    print(f"inv_cov shape: {inv_cov.shape}")
    
    # Intentar multiplicación paso a paso
    try:
        step1 = x_i.T
        print(f"step1 (x_i.T) shape: {step1.shape}")
        
        step2 = step1 @ inv_cov
        print(f"step2 (x_i.T @ inv_cov) shape: {step2.shape}")
        
        step3 = step2 @ x_i
        print(f"step3 (step2 @ x_i) shape: {step3.shape}")
        
        print("Multiplicación exitosa")
        
    except Exception as e:
        print(f"Error en multiplicación: {e}")
        print(f"Detalles del error:")
        print(f"x_i.T shape: {x_i.T.shape}")
        print(f"inv_cov shape: {inv_cov.shape}")
        print(f"x_i shape: {x_i.shape}")

if __name__ == "__main__":
    debug_faster_qda() 