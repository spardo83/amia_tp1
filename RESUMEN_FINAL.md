# TP 2: LDA/QDA y optimización matemática - RESUMEN FINAL

    
    ## Integrantes
    
    - Pardo, Sebastián
    - González, Martín
    - Brazón, Josmar
    - Losada, Ricardo
  


##  Implementaciones Completadas

Todas las implementaciones requeridas han sido completadas y funcionan correctamente:

### 1. **Tensorización** (Preguntas 1-7)
-  `QDA`: Implementación estándar
-  `TensorizedQDA`: Paraleliza sobre clases
-  `FasterQDA`: Elimina ciclo for en predicción
-  `EfficientQDA`: Usa propiedad matemática para evitar matriz n×n

### 2. **Factorización de Cholesky** (Preguntas 8-13)
-  `QDA_Chol1`: Usa `LA.inv(cholesky(...))`
-  `QDA_Chol2`: Usa `solve_triangular` (más eficiente)
-  `QDA_Chol3`: Usa `dtrtri` (LAPACK)
-  `TensorizedChol`: Combina tensorización con Cholesky
-  `EfficientChol`: Implementación más optimizada

##  Resultados de Testing

Todas las implementaciones mantienen la misma precisión:
- **Accuracy**: 0.981 (98.1%)
- **Dataset**: Wine (178 observaciones, 13 features, 3 clases)
- **Split**: 70% train, 30% test

## 🔧 Archivos del Proyecto

### Implementaciones
- `implementations_final.py`: Todas las implementaciones corregidas y funcionales
- `TP1_original.ipynb`: Notebook original del TP

### Testing y Debug
- `test_final.py`: Script de prueba final
- `debug_faster_qda.py`: Debug específico para FasterQDA
- `debug_test.py`: Debug general de implementaciones

### Documentación
- `respuestas_teoricas.md`: Respuestas a todas las preguntas teóricas
- `README.md`: Instrucciones de uso
- `requirements.txt`: Dependencias

## Respuestas a Preguntas Clave

### Pregunta 1: ¿Sobre qué paraleliza TensorizedQDA?
**Respuesta**: TensorizedQDA paraleliza sobre las **k clases**, no sobre las n observaciones. Usa operaciones tensoriales para calcular la forma cuadrática para todas las clases simultáneamente.

### Pregunta 4: ¿Dónde aparece la matriz n×n?
**Respuesta**: En `FasterQDA`, la matriz n×n aparece en el producto `unbiased_x.T @ inv_cov @ unbiased_x` cuando `unbiased_x` tiene shape `(p, n)`. Esta matriz contiene todas las interacciones entre observaciones.

### Pregunta 5: Demostración matemática
**Respuesta**: Se demuestra que `diag(A·B) = sum(A ⊙ B^T, axis=1)`, donde ⊙ es la multiplicación elemento por elemento. Esto permite calcular solo la diagonal sin construir la matriz completa n×n.

### Pregunta 8: A^(-1) en términos de L
**Respuesta**: Si A = LL^T, entonces A^(-1) = L^(-T) · L^(-1). Esto es útil porque:
`(x-μ)^T Σ^(-1) (x-μ) = ||L^(-1)(x-μ)||^2`

### Pregunta 9: Diferencias entre implementaciones Cholesky
**Respuesta**:
- **Chol1**: Calcula L^(-1) explícitamente
- **Chol2**: Usa `solve_triangular` (más eficiente)
- **Chol3**: Usa LAPACK para L^(-1)

## Cómo Usar

### 1. Instalar dependencias
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Probar implementaciones
```bash
python test_final.py
```

### 3. Usar en código
```python
from implementations_final import *

# Cargar datos
X_full, y_full = get_wine_dataset()
y_full_encoded = label_encode(y_full)

# Probar implementación
model = EfficientQDA()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Optimizaciones Implementadas

### 1. **Tensorización**
- Usar `np.stack()` para paralelizar sobre clases
- Operaciones tensoriales para calcular múltiples clases simultáneamente

### 2. **Eliminación de Bucles**
- Vectorizar operaciones cuando sea posible
- Reducir iteraciones innecesarias

### 3. **Propiedades Matemáticas**
- Usar `diag(A·B) = sum(A ⊙ B^T, axis=1)` para evitar matrices n×n
- Calcular solo la diagonal sin construir la matriz completa

### 4. **Factorización de Cholesky**
- Usar `solve_triangular` en lugar de inversión
- Aprovechar la estructura triangular de L

## Conceptos Clave Aprendidos

1. **Clasificación Bayesiana**: Entender la regla de decisión de Bayes
2. **Forma Cuadrática**: Optimizar el cálculo de `(x-μ)^T Σ^(-1) (x-μ)`
3. **Tensorización**: Paralelizar operaciones usando tensores
4. **Factorización de Cholesky**: Usar LL^T para optimizar inversiones
5. **Optimización Numérica**: Balancear precisión, velocidad y memoria

## Análisis de Performance

Basándome en la teoría y las implementaciones:

1. **Tensorización**: TensorizedQDA debería ser más rápido en predicción
2. **Optimización**: EfficientQDA debería ser más eficiente en memoria
3. **Cholesky**: Chol2 debería ser el más eficiente por usar `solve_triangular`
4. **Combinaciones**: EfficientChol debería ser la implementación más optimizada

## Versiones Utilizadas

- **Python**: 3.13.5
- **NumPy**: 2.3.2
- **SciPy**: 1.16.1
- **scikit-learn**: 1.7.1
- **pandas**: 2.3.1
- **tqdm**: 4.67.1

