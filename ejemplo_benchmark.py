#!/usr/bin/env python3
"""
Script de ejemplo para probar y hacer benchmark de todas las implementaciones del TP
"""

import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_wine
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Importar nuestras implementaciones
from implementations import *

def get_wine_dataset():
    """Cargar dataset de vino"""
    data = load_wine()
    X_full = data.data
    y_full = np.array([data.target_names[y] for y in data.target.reshape(-1,1)])
    return X_full, y_full

def label_encode(y_full):
    """Codificar labels a números"""
    return LabelEncoder().fit_transform(y_full.flatten()).reshape(y_full.shape)

def split_transpose(X, y, test_size, random_state):
    """Split y transponer datos"""
    return [elem.T for elem in train_test_split(X, y, test_size=test_size, random_state=random_state)]

class SimpleBenchmark:
    """Benchmark simplificado para probar las implementaciones"""
    
    def __init__(self, X, y, n_runs=10, test_sz=0.3, random_state=42):
        self.X = X
        self.y = y
        self.n_runs = n_runs
        self.test_sz = test_sz
        self.random_state = random_state
        self.results = {}
    
    def bench_model(self, model_class, name):
        """Benchmark de un modelo específico"""
        print(f"Benchmarking {name}...")
        
        train_times = []
        test_times = []
        accuracies = []
        
        for i in tqdm(range(self.n_runs), desc=name):
            # Split de datos
            X_train, X_test, y_train, y_test = split_transpose(
                self.X, self.y, test_size=self.test_sz, random_state=self.random_state + i
            )
            
            # Entrenamiento
            model = model_class()
            t1 = time.perf_counter()
            model.fit(X_train, y_train)
            t2 = time.perf_counter()
            train_time = (t2 - t1) * 1000  # ms
            
            # Predicción
            t3 = time.perf_counter()
            preds = model.predict(X_test)
            t4 = time.perf_counter()
            test_time = (t4 - t3) * 1000  # ms
            
            # Accuracy
            accuracy = (y_test.flatten() == preds.flatten()).mean()
            
            train_times.append(train_time)
            test_times.append(test_time)
            accuracies.append(accuracy)
        
        self.results[name] = {
            'train_median_ms': np.median(train_times),
            'train_std_ms': np.std(train_times),
            'test_median_ms': np.median(test_times),
            'test_std_ms': np.std(test_times),
            'mean_accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies)
        }
    
    def run_all_benchmarks(self):
        """Ejecutar benchmark de todos los modelos"""
        models = [
            (QDA, "QDA"),
            (TensorizedQDA, "TensorizedQDA"),
            (FasterQDA, "FasterQDA"),
            (EfficientQDA, "EfficientQDA"),
            (QDA_Chol1, "QDA_Chol1"),
            (QDA_Chol2, "QDA_Chol2"),
            (QDA_Chol3, "QDA_Chol3"),
            (TensorizedChol, "TensorizedChol"),
            (EfficientChol, "EfficientChol")
        ]
        
        for model_class, name in models:
            try:
                self.bench_model(model_class, name)
            except Exception as e:
                print(f"Error en {name}: {e}")
                self.results[name] = {
                    'train_median_ms': np.nan,
                    'train_std_ms': np.nan,
                    'test_median_ms': np.nan,
                    'test_std_ms': np.nan,
                    'mean_accuracy': np.nan,
                    'accuracy_std': np.nan
                }
    
    def print_results(self, baseline='QDA'):
        """Imprimir resultados del benchmark"""
        if not self.results:
            print("No hay resultados para mostrar")
            return
        
        df = pd.DataFrame(self.results).T
        
        # Calcular speedups si hay baseline
        if baseline in self.results:
            baseline_train = self.results[baseline]['train_median_ms']
            baseline_test = self.results[baseline]['test_median_ms']
            
            df['train_speedup'] = baseline_train / df['train_median_ms']
            df['test_speedup'] = baseline_test / df['test_median_ms']
        
        print("\n" + "="*80)
        print("RESULTADOS DEL BENCHMARK")
        print("="*80)
        
        # Mostrar métricas principales
        cols = ['train_median_ms', 'test_median_ms', 'mean_accuracy']
        if baseline in self.results:
            cols.extend(['train_speedup', 'test_speedup'])
        
        print(df[cols].round(3))
        
        # Análisis de performance
        print("\n" + "="*80)
        print("ANÁLISIS DE PERFORMANCE")
        print("="*80)
        
        if baseline in self.results:
            best_train = df['train_speedup'].idxmax()
            best_test = df['test_speedup'].idxmax()
            best_acc = df['mean_accuracy'].idxmax()
            
            print(f"Mejor en entrenamiento: {best_train} ({df.loc[best_train, 'train_speedup']:.2f}x más rápido)")
            print(f"Mejor en predicción: {best_test} ({df.loc[best_test, 'test_speedup']:.2f}x más rápido)")
            print(f"Mejor accuracy: {best_acc} ({df.loc[best_acc, 'mean_accuracy']:.3f})")

def test_implementations():
    """Función principal para probar las implementaciones"""
    print("Cargando dataset...")
    X_full, y_full = get_wine_dataset()
    y_full_encoded = label_encode(y_full)
    
    print(f"Dataset shape: {X_full.shape}")
    print(f"Clases: {np.unique(y_full_encoded)}")
    
    # Crear benchmark
    benchmark = SimpleBenchmark(
        X_full, y_full_encoded,
        n_runs=20,  # Pocos runs para prueba rápida
        test_sz=0.3,
        random_state=42
    )
    
    # Ejecutar benchmarks
    benchmark.run_all_benchmarks()
    
    # Mostrar resultados
    benchmark.print_results(baseline='QDA')
    
    return benchmark

if __name__ == "__main__":
    # Verificar versiones
    import sys
    import numpy as np
    import scipy
    
    print("Versiones utilizadas:")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print()
    
    # Ejecutar benchmark
    benchmark = test_implementations() 