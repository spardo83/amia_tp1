import numpy as np
import numpy.linalg as LA
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg.lapack import dtrtri

from base.bayesian import BaseBayesianClassifier


class QDA_Chol1(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.L_invs = [
        LA.inv(cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True))
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L_inv = self.L_invs[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = L_inv @ unbiased_x

    return np.log(L_inv.diagonal().prod()) -0.5 * (y**2).sum()


class QDA_Chol2(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.Ls = [
        cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True)
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L = self.Ls[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = solve_triangular(L, unbiased_x, lower=True)

    return -np.log(L.diagonal().prod()) -0.5 * (y**2).sum()


class QDA_Chol3(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.L_invs = [
        dtrtri(cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True), lower=1)[0]
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L_inv = self.L_invs[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = L_inv @ unbiased_x

    return np.log(L_inv.diagonal().prod()) -0.5 * (y**2).sum()
  
class TensorizedChol(QDA_Chol3):
  """
  Tensoriza solo en clases
  """

  def _fit_params(self, X, y):
      
      super()._fit_params(X, y)  

      # Apilar para obtener tensores (K, d, d) y (K, d, 1)

      self.L_invs = np.stack(self.L_invs, axis=0)    # (K, d, d)
      self.means  = np.stack(self.means,  axis=0)    # (K, d, 1)

      # Precomputar determinantes
      
      self.log_diag_Linv_prod = np.log(
          np.diagonal(self.L_invs, axis1=1, axis2=2)
      ).sum(axis=1)                                   # (K,)

  def _predict_log_conditional(self, x, class_idx):
      
      L_inv = self.L_invs[class_idx]                 # (d, d)
      mu    = self.means[class_idx]                  # (d, 1)
      y     = L_inv @ (x - mu)                       # (d, 1)

      return float(self.log_diag_Linv_prod[class_idx] - 0.5 * (y * y).sum())

class EfficientChol(TensorizedChol):
    """
    Igual a FastQDA en espíritu (vectorizar en K y n).
    Calcula directamente:
        y_{k,i} = L_k^{-1} (x_i - mu_k)
        quad_{k,i} = ||y_{k,i}||^2
    y usa log_norm[k] = sum(log diag(L_k^{-1})) = -0.5 log|Sigma_k|.
    """

    def _predict_log_conditionals_batch(self, X):
        """
        X: (d, n)  → columnas = observaciones
        Return: (K, n) con log P(x_i | G=k) hasta constante común.
        """
        # U[k] = X - mu_k  → (K, d, n)
        U = X[None, :, :] - self.means               # broadcast en K

        # y[k] = L_inv[k] @ U[k]  → (K, d, n)  (matmul batcheado)
        y = self.L_invs @ U

        # quad[k, i] = sum_d y[k, d, i]^2  → (K, n)
        quad = np.sum(y * y, axis=1)

        # log-condicional por clase y observación
        return self.log_diag_Linv_prod[:, None] - 0.5 * quad   # (K, n)

    def predict(self, X):
        """
        Predice clase por observación, vectorizado en K y n.
        """
        log_post = self.log_a_priori[:, None] + self._predict_log_conditionals_batch(X)  # (K, n)
        y_hat = np.argmax(log_post, axis=0)                                              # (n,)
        return y_hat.reshape(1, -1)

  
  
