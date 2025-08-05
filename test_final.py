#!/usr/bin/env python3
"""
Script final para probar todas las implementaciones corregidas
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

def test_all_implementations():
    """Probar todas las implementaciones"""
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
    
    # Lista de modelos a probar
    models = [
        ("QDA", QDA),
        ("TensorizedQDA", TensorizedQDA),
        ("FasterQDA", FasterQDA),
        ("EfficientQDA", EfficientQDA),
        ("QDA_Chol1", QDA_Chol1),
        ("QDA_Chol2", QDA_Chol2),
        ("QDA_Chol3", QDA_Chol3),
        ("TensorizedChol", TensorizedChol),
        ("EfficientChol", EfficientChol)
    ]
    
    results = {}
    
    for name, model_class in models:
        print(f"\nProbando {name}...")
        try:
            model = model_class()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accuracy = (y_test.flatten() == preds.flatten()).mean()
            results[name] = accuracy
            print(f"{name} accuracy: {accuracy:.3f}")
        except Exception as e:
            print(f"Error en {name}: {e}")
            results[name] = None
    
    print("\n" + "="*50)
    print("RESUMEN DE RESULTADOS")
    print("="*50)
    for name, accuracy in results.items():
        if accuracy is not None:
            print(f"{name}: {accuracy:.3f}")
        else:
            print(f"{name}: ERROR")

if __name__ == "__main__":
    test_all_implementations() 