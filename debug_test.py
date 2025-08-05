#!/usr/bin/env python3
"""
Script para debuggear los errores específicos
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Importar implementaciones corregidas
from implementations_fixed import *

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
    """Debuggear FasterQDA"""
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
    
    # Probar con una observación
    x = X_test[:, 0:1]  # Shape: (13, 1)
    print(f"x shape: {x.shape}")
    
    # Calcular unbiased_x para clase 0
    class_idx = 0
    unbiased_x = x - model.tensor_means[class_idx:class_idx+1, :, :]
    print(f"unbiased_x shape: {unbiased_x.shape}")
    
    # Intentar multiplicación
    try:
        result = unbiased_x.T @ model.tensor_inv_cov[class_idx] @ unbiased_x
        print(f"Result shape: {result.shape}")
        print("Multiplicación exitosa")
    except Exception as e:
        print(f"Error en multiplicación: {e}")

def debug_efficient_qda():
    """Debuggear EfficientQDA"""
    print("\n=== Debug EfficientQDA ===")
    
    # Cargar datos
    X_full, y_full = get_wine_dataset()
    y_full_encoded = label_encode(y_full)
    X_train, X_test, y_train, y_test = split_transpose(
        X_full, y_full_encoded, test_size=0.3, random_state=42
    )
    
    # Entrenar modelo
    model = EfficientQDA()
    model.fit(X_train, y_train)
    
    print(f"X_test shape: {X_test.shape}")
    
    # Probar con una clase
    class_idx = 0
    unbiased_x = X_test - model.tensor_means[class_idx:class_idx+1, :, :]
    inv_cov = model.tensor_inv_cov[class_idx]
    
    print(f"unbiased_x shape: {unbiased_x.shape}")
    print(f"inv_cov shape: {inv_cov.shape}")
    
    # Calcular A y B
    A = unbiased_x.T  # Shape: (n, p)
    B = inv_cov @ unbiased_x  # Shape: (p, n)
    B_T = B.T  # Shape: (n, p)
    
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print(f"B_T shape: {B_T.shape}")
    
    # Intentar multiplicación elemento por elemento
    try:
        element_wise_prod = A * B_T
        print(f"element_wise_prod shape: {element_wise_prod.shape}")
        quadratic_forms = np.sum(element_wise_prod, axis=1)
        print(f"quadratic_forms shape: {quadratic_forms.shape}")
        print("Cálculo exitoso")
    except Exception as e:
        print(f"Error en cálculo: {e}")

def debug_efficient_chol():
    """Debuggear EfficientChol"""
    print("\n=== Debug EfficientChol ===")
    
    # Cargar datos
    X_full, y_full = get_wine_dataset()
    y_full_encoded = label_encode(y_full)
    X_train, X_test, y_train, y_test = split_transpose(
        X_full, y_full_encoded, test_size=0.3, random_state=42
    )
    
    # Entrenar modelo
    model = EfficientChol()
    model.fit(X_train, y_train)
    
    print(f"X_test shape: {X_test.shape}")
    
    # Probar con una clase
    class_idx = 0
    unbiased_x = X_test - model.tensor_means[class_idx:class_idx+1, :, :]
    L = model.tensor_Ls[class_idx]
    
    print(f"unbiased_x shape: {unbiased_x.shape}")
    print(f"L shape: {L.shape}")
    
    # Intentar resolver sistema triangular
    try:
        y_solutions = solve_triangular(L, unbiased_x, lower=True)
        print(f"y_solutions shape: {y_solutions.shape}")
        quadratic_forms = np.sum(y_solutions**2, axis=0)
        print(f"quadratic_forms shape: {quadratic_forms.shape}")
        print("Cálculo exitoso")
    except Exception as e:
        print(f"Error en cálculo: {e}")

if __name__ == "__main__":
    debug_faster_qda()
    debug_efficient_qda()
    debug_efficient_chol() 