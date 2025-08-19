import numpy as np
import numpy.linalg as LA

from base.bayesian import BaseBayesianClassifier


class QDA(BaseBayesianClassifier):

  def _fit_params(self, X, y):
    # estimate each covariance matrix
    self.inv_covs = [LA.inv(np.cov(X[:,y.flatten()==idx], bias=True))
                      for idx in range(len(self.log_a_priori))]
    # Q5: por que hace falta el flatten y no se puede directamente X[:,y==idx]?
    # Q6: por que se usa bias=True en vez del default bias=False?
    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]
    # Q7: que hace axis=1? por que no axis=0?

  def _predict_log_conditional(self, x, class_idx):
    # predict the log(P(x|G=class_idx)), the log of the conditional probability of x given the class
    # this should depend on the model used
    inv_cov = self.inv_covs[class_idx]
    unbiased_x =  x - self.means[class_idx]
    return 0.5*np.log(LA.det(inv_cov)) -0.5 * unbiased_x.T @ inv_cov @ unbiased_x


class TensorizedQDA(QDA):

    def _fit_params(self, X, y):
        # ask plain QDA to fit params
        super()._fit_params(X,y)

        # stack onto new dimension
        self.tensor_inv_cov = np.stack(self.inv_covs)
        self.tensor_means = np.stack(self.means)

    def _predict_log_conditionals(self,x):
        unbiased_x = x - self.tensor_means
        inner_prod = unbiased_x.transpose(0,2,1) @ self.tensor_inv_cov @ unbiased_x

        return 0.5*np.log(LA.det(self.tensor_inv_cov)) - 0.5 * inner_prod.flatten()

    def _predict_one(self, x):
        # return the class that has maximum a posteriori probability
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))

class FastQDA(TensorizedQDA):
    """
    Variante de QDA que vectoriza tanto sobre las clases (k) como sobre las
    observaciones (n), construyendo explícitamente, para cada clase k, la matriz
    de interacciones entre observaciones:

        M_k = (X - μ_k)^T  Σ_k^{-1}  (X - μ_k)   ∈ R^{n×n}
    
    De esa matriz solo se necesita su diagonal para los términos cuadráticos
    individuales de cada observación.

    Notas
    -----
    - Esta implementación es correcta pero costosa en memoria: M tiene shape
      (k, n, n) → O(k·n²).
    """

    def _predict_log_conditionals_batch(self, x):
        """
        Calcula en paralelo (sobre k y n) los log-condicionales devolviendo una matriz (k, n).

        Notas:
        1) Se desplazan todas las observaciones por la media de cada clase:
               U[k] = X - μ_k     → U.shape = (k, p, n)
        2) Se construye la matriz n×n por clase:
               M[k] = U[k]^T Σ_k^{-1} U[k]   → M.shape = (k, n, n)
        3) Se extrae la diagonal por clase (término cuadrático por observación):
               quad[k, i] = (x_i-μ_k)^T Σ_k^{-1} (x_i-μ_k)  → quad.shape = (k, n)
        4) Se suma el término 0.5·log|Σ_k^{-1}| (expandido a (k,1)) y se resta
           0.5·quad para obtener (k, n).
        """
        # U[k] = X - μ_k  → (k, p, n)  (broadcasting en eje de clases)
        U = x - self.tensor_means

        # M = U^T Σ_k^{-1} U → (k, n, n)  (matmul batcheado sobre k)
        M = U.transpose(0, 2, 1) @ self.tensor_inv_cov @ U

        # Diagonal por clase: q_{k,i} = (x_i-μ_k)^T Σ_k^{-1} (x_i-μ_k) → (k, n)
        quad = np.diagonal(M, axis1=1, axis2=2)

        # 0.5·log|Σ_k^{-1}| → (k,)  ⇒ (k,1) para broadcast sobre n
        logdet = 0.5 * np.log(LA.det(self.tensor_inv_cov)).reshape(-1, 1)

        # Log-condicional
        return logdet - 0.5 * quad  # (k, n)

    def predict(self, X):
        """
        Predice la clase por observación sin bucles.

        Notas:
        - Se suma self.log_a_priori[:, None] (shape (k,1)) para que haga
          broadcasting correcto contra (k, n).
        """
        # Log-posteriori = log a priori + log-condicional  → (k, n)
        log_post = self.log_a_priori[:, None] + self._predict_log_conditionals_batch(X)

        # Clase de máximo a posteriori por columna
        y_hat = np.argmax(log_post, axis=0)  # (n,)

        return y_hat.reshape(1, -1)

    def predict_with_probs(self, X):
        """
        Predice la clase y devuelve también las probabilidades a posteriori por clase
        (softmax de los log-posteriori), todo vectorizado y sin bucles.

        Returns
        -------
        out : ndarray de shape (n, 1 + k)
            - Columna 0: clase predicha (como float para mantener compatibilidad con la API original).
            - Columnas 1..k: probabilidades a posteriori por clase para cada observación.

        Detalles
        --------
        1) Se calculan log-posteriori (k, n).
        2) Se aplica softmax columna a columna de manera estable:
               probs = exp(log_post - max_col) / sum(exp(...))
        3) Se empaqueta la salida en (n, 1 + k).
        """
        # 1) Log-posteriori: (k, n)
        log_post = self.log_a_priori[:, None] + self._predict_log_conditionals_batch(X)

        # 2) Clase por observación
        y_hat = np.argmax(log_post, axis=0)  # (n,)

        # 3) Softmax estable por columnas (sobre eje de clases)
        max_per_col = np.max(log_post, axis=0, keepdims=True)                # (1, n)
        exp_shifted = np.exp(log_post - max_per_col)                         # (k, n)
        probs = exp_shifted / np.sum(exp_shifted, axis=0, keepdims=True)     # (k, n)

        # 4) Empaquetar salida
        n = X.shape[1]
        k = log_post.shape[0]
        out = np.empty((n, 1 + k), dtype=float)
        out[:, 0] = y_hat.astype(float)  # primera columna = clase predicha
        out[:, 1:] = probs.T             # (n, k)

        return out
    
class EfficientQDA(FastQDA):
  """
  Igual a FastQDA pero evita construir la matriz (n x n).
  Calcula directamente la diagonal aprovechando que diag(A B) = sum_cols( A ⊙ B^T ).
  A = U^T Σ_k^{-1}
  B = U^T
  """

  def _predict_log_conditionasls_batch(self, X):
      """
      X: (p, n)  -> columnas = observaciones
      Return: (k, n) con log P(x_i | G=k) hasta constante común.
      """
      # U[k] = X - μ_k  → (k, p, n)
      U = X[None, :, :] - self.tensor_means
      U_T = U.transpose(0, 2, 1)
      
      # quad[k, i] = (x_i-μ_k)^T Σ_k^{-1} (x_i-μ_k)  → (k, n)
      # usando sum_cols(A ⊙ B^T)
      quad = np.sum(U_T @ self.tensor_inv_cov * U_T, axis=2)
      
      # 0.5·log|Σ_k^{-1}| → (k,)  ⇒ (k,1) para broadcast sobre n
      logdet = 0.5 * np.log(LA.det(self.tensor_inv_cov)).reshape(-1, 1)

      # Log-condicional
      return logdet - 0.5 * quad  # (k, n)