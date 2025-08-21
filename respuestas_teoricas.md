# TP 1: LDA/QDA y optimización matemática

Integrantes:

- Brazón, Josmar

- Pardo, Sebastián

- González, Martín

- Losada, Ricardo

## Respuestas Teóricas 

## Pregunta 1: Diferencias entre QDA y TensorizedQDA

**TensorizedQDA paraleliza sobre las k clases**, no sobre las n observaciones. Esto se puede ver en:

- `tensor_inv_cov` tiene shape `(k, p, p)` - una matriz de covarianza inversa por clase
- `tensor_means` tiene shape `(k, p, 1)` - un vector de medias por clase
- En `_predict_log_conditionals`, se calcula `unbiased_x = x - self.tensor_means` que tiene shape `(k, p, 1)`
- El producto `unbiased_x.transpose(0,2,1) @ self.tensor_inv_cov @ unbiased_x` calcula la forma cuadrática para todas las clases de una vez

**Paso a paso cómo llega al mismo resultado que QDA:**
1. Para cada clase j, calcula `x - μ_j` (centrado)
2. Calcula `(x - μ_j)^T Σ_j^(-1) (x - μ_j)` (forma cuadrática)
3. Aplica la fórmula logarítmica: `0.5*log|Σ_j^(-1)| - 0.5 * forma_cuadrática`
4. Retorna el argmax de los log posteriores

## Pregunta 2: Sobre qué paraleliza TensorizedQDA

TensorizedQDA paraleliza sobre las **k clases**, no sobre las n observaciones. Esto significa que:

- En lugar de calcular la forma cuadrática para una clase a la vez, lo hace para todas las clases simultáneamente
- Los shapes de `tensor_inv_covs` y `tensor_means` son `(k, p, p)` y `(k, p, 1)` respectivamente
- El método `_predict_log_conditionals` calcula los log posteriores para todas las clases de una vez usando operaciones tensoriales

## Pregunta 4: Matriz n×n en FasterQDA

En `FasterQDA`, la matriz n×n aparece cuando intentamos calcular la forma cuadrática para todas las observaciones de una vez:

```python
# Si X tiene shape (p, n) y queremos calcular (X-μ)^T Σ^(-1) (X-μ)
# Esto resulta en una matriz de shape (n, n) que contiene todas las interacciones
# entre observaciones, pero solo necesitamos la diagonal
```

La matriz n×n aparece en el producto `unbiased_x.T @ inv_cov @ unbiased_x` cuando `unbiased_x` tiene shape `(p, n)`.

## Pregunta 5: Demostración de la propiedad

**Demostración:**

Sea A una matriz de shape (n, p) y B una matriz de shape (p, n).

El elemento (i,j) de A·B es: `(A·B)_{i,j} = Σ_{k=1}^p A_{i,k} · B_{k,j}`

La diagonal de A·B es: `diag(A·B)_i = (A·B)_{i,i} = Σ_{k=1}^p A_{i,k} · B_{k,i}`

Por otro lado, `(A ⊙ B^T)_{i,k} = A_{i,k} · B^T_{i,k} = A_{i,k} · B_{k,i}`

Entonces: `sum(A ⊙ B^T, axis=1)_i = Σ_{k=1}^p (A ⊙ B^T)_{i,k} = Σ_{k=1}^p A_{i,k} · B_{k,i} = diag(A·B)_i`

**QED**

## Pregunta 7: Diferencias entre QDA_Chol1 y QDA

**QDA_Chol1** usa la factorización de Cholesky para optimizar el cálculo de la forma cuadrática:

1. **Entrenamiento**: Calcula L^(-1) directamente usando `LA.inv(cholesky(...))`
2. **Predicción**: 
   - Calcula `unbiased_x = x - μ_j`
   - Resuelve `y = L^(-1) @ unbiased_x`
   - Calcula `||y||^2 = Σ y_i^2`
   - Log det: `log(L_inv.diagonal().prod())`

**Ventajas sobre QDA:**
- Evita calcular la inversa completa de Σ
- Usa la estructura triangular de L^(-1)
- Más eficiente para matrices grandes

## Pregunta 8: A^(-1) en términos de L

Si A = LL^T, entonces A^(-1) = (LL^T)^(-1) = (L^T)^(-1) · L^(-1) = L^(-T) · L^(-1)

Esto es útil en la forma cuadrática de QDA porque:

`(x-μ)^T Σ^(-1) (x-μ) = (x-μ)^T L^(-T) L^(-1) (x-μ) = (L^(-1)(x-μ))^T (L^(-1)(x-μ)) = ||L^(-1)(x-μ)||^2`

Donde ||·|| es la norma euclidiana. Esto es más eficiente porque:
1. L^(-1) es triangular, por lo que resolver L^(-1)y es más rápido que invertir Σ
2. Calcular ||y||^2 es más rápido que calcular la forma cuadrática completa

## Pregunta 9: Diferencias entre implementaciones Cholesky

**QDA_Chol1:**
- Calcula L^(-1) directamente usando `LA.inv(cholesky(...))`
- En predicción: `y = L_inv @ unbiased_x`
- Log det: `log(L_inv.diagonal().prod())`

**QDA_Chol2:**
- Guarda L directamente usando `cholesky(...)`
- En predicción: `y = solve_triangular(L, unbiased_x, lower=True)`
- Log det: `-log(L.diagonal().prod())`

**QDA_Chol3:**
- Usa `dtrtri` (LAPACK) para calcular L^(-1) más eficientemente
- En predicción: `y = L_inv @ unbiased_x`
- Log det: `log(L_inv.diagonal().prod())`

**Diferencias clave:**
- Chol1 y Chol3 calculan L^(-1), Chol2 guarda L
- Chol2 usa `solve_triangular` que es más eficiente que multiplicar por L^(-1)
- Chol3 usa LAPACK que puede ser más rápido para matrices grandes

## Pregunta 10: Comparación de performance

Basándome en la teoría y las implementaciones:

**Tensorización (QDA vs TensorizedQDA):**
- TensorizedQDA debería ser más rápido en predicción porque paraleliza sobre clases
- El entrenamiento debería ser similar o ligeramente más lento

**Optimización (FasterQDA vs EfficientQDA):**
- EfficientQDA debería ser más eficiente en memoria porque evita la matriz n×n
- FasterQDA puede ser más rápido pero usa más memoria

**Cholesky:**
- Chol2 debería ser el más eficiente porque usa `solve_triangular`
- Chol3 puede ser más rápido para matrices grandes por usar LAPACK
- Chol1 es el menos eficiente porque calcula L^(-1) explícitamente

**Combinaciones:**
- TensorizedChol debería combinar las ventajas de tensorización y Cholesky
- EfficientChol debería ser la implementación más optimizada en general 