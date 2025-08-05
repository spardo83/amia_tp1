#!/usr/bin/env python3
"""
Script simple para probar las implementaciones
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Importar nuestras implementaciones
from implementations import *

def get_wine_dataset():
    data = load_wine()
    X_full = data.data
    y_full = np.array([data.target_names[y] for y in data.target.reshape(-1,1)])
    return X_full, y_full

def label_encode(y_full):
    return LabelEncoder().fit_transform(y_full.flatten()).reshape(y_full.shape)

def split_transpose(X, y, test_size, random_state):
    return [elem.T for elem in train_test_split(X, y, test_size=test_size, random_state=random_state)]

def test_basic_functionality():
    """Probar funcionalidad básica"""
    print("Cargando datos...")
    X_full, y_full = get_wine_dataset()
    y_full_encoded = label_encode(y_full)
    
    print(f"Dataset shape: {X_full.shape}")
    print(f"Clases: {np.unique(y_full_encoded)}")
    
    # Split de datos
    X_train, X_test, y_train, y_test = split_transpose(
        X_full, y_full_encoded, test_size=0.3, random_state=42
    )
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Probar QDA básico
    print("\nProbando QDA...")
    model = QDA()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = (y_test.flatten() == preds.flatten()).mean()
    print(f"QDA accuracy: {accuracy:.3f}")
    
    # Probar TensorizedQDA
    print("\nProbando TensorizedQDA...")
    model = TensorizedQDA()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = (y_test.flatten() == preds.flatten()).mean()
    print(f"TensorizedQDA accuracy: {accuracy:.3f}")
    
    # Probar FasterQDA
    print("\nProbando FasterQDA...")
    try:
        model = FasterQDA()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = (y_test.flatten() == preds.flatten()).mean()
        print(f"FasterQDA accuracy: {accuracy:.3f}")
    except Exception as e:
        print(f"Error en FasterQDA: {e}")
    
    # Probar EfficientQDA
    print("\nProbando EfficientQDA...")
    try:
        model = EfficientQDA()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = (y_test.flatten() == preds.flatten()).mean()
        print(f"EfficientQDA accuracy: {accuracy:.3f}")
    except Exception as e:
        print(f"Error en EfficientQDA: {e}")
    
    # Probar Cholesky
    print("\nProbando QDA_Chol2...")
    model = QDA_Chol2()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = (y_test.flatten() == preds.flatten()).mean()
    print(f"QDA_Chol2 accuracy: {accuracy:.3f}")
    
    # Probar EfficientChol
    print("\nProbando EfficientChol...")
    try:
        model = EfficientChol()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = (y_test.flatten() == preds.flatten()).mean()
        print(f"EfficientChol accuracy: {accuracy:.3f}")
    except Exception as e:
        print(f"Error en EfficientChol: {e}")

if __name__ == "__main__":
    test_basic_functionality() 