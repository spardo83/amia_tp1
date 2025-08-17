import numpy as np
import numpy.linalg as LA

from base.bayesian import BaseBayesianClassifier

class QDA(BaseBayesianClassifier):
  """
    Quadratic Discriminant Analysis (QDA).

    Esta subclase implementa un clasificador bayesiano con verosimilitud
    gaussiana multivariada y covarianzas específicas por clase:
        X | (G=j) ~ N( mu_j, Sigma_j )

    Parámetros/convenciones de forma:
    - X: ndarray de forma (n_features, n_obs). Cada columna es una observación.
    - y: ndarray con etiquetas enteras en {0, 1, ..., K-1}. Se asume forma (1, n_obs) o equivalente.
    - self.log_a_priori: vector con log π_j (lo calcula BaseBayesianClassifier.fit).

    Entrenamiento:
    - _fit_params estima (por clase) la media mu_j y la inversa de la covarianza Sigma_j^{-1}.
      Se usa el estimador MLE (normalización por N) para la covarianza (bias=True).

    Predicción:
    - _predict_log_conditional devuelve log f_{X|G=j}(x) (hasta una constante común),
      usando la forma cuadrática con la matriz de precisión (inversa de la covarianza).
    """
  def _fit_params(self, X, y):
    """
    Estima, para cada clase j:
      - self.inv_covs[j] = (Sigma_j)^{-1}  (inversa de la covarianza por clase)
      - self.means[j]    = mu_j            (media por clase, como vector columna de forma (n_features, 1))
    """

    self.inv_covs = [LA.inv(np.cov(X[:,y.flatten()==idx], bias=True))
                      for idx in range(len(self.log_a_priori))]

    print(f"Cantidad de Inversas de la covarianza: {len(self.inv_covs)} [Formato: {self.inv_covs[0].shape}]")

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]
    
    print(f"Cantidad de medias: {len(self.means)} [Formato: {self.means[0].shape}]")

  def _predict_log_conditional(self, x, class_idx):
    """
    Devuelve el logaritmo de la densidad condicional (hasta constante) para la clase `class_idx`:

        log f_{X|G=j}(x) = -0.5 * (x - mu_j)^T Sigma_j^{-1} (x - mu_j)  - 0.5 * log |Sigma_j|  + C

    Observación:
    - Como se almacena inv_cov = Sigma_j^{-1}, y se usa 0.5 * log(det(inv_cov)) = -0.5 * log |Sigma_j|,
      la expresión implementada es algebraicamente equivalente (hasta la constante C común a todas las clases).

    Parámetros:
    - x : ndarray de forma (n_features, 1) (una sola observación como vector columna)
    - class_idx : int índice de clase j

    Devuelve:
    - escalar (numpy) con el valor de log f_{X|G=class_idx}(x) + C
    """
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
