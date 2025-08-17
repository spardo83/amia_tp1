import numpy as np

class BaseBayesianClassifier:
  def __init__(self):
    """
    Constructor de la clase base para clasificadores bayesianos.
    """
    pass

  def _estimate_a_priori(self, y):
    """
    Estima las probabilidades a priori de cada clase a partir del vector de etiquetas `y`.

    Parámetros:
    -----------
    y : array-like, forma (n_obs,) o similar
        Vector de etiquetas con valores enteros no negativos (0, 1, ..., k-1).

    Devuelve:
    ---------
    log_a_priori : ndarray
        Logaritmo natural de las probabilidades a priori para cada clase.

    Notas:
    ------
    - np.bincount cuenta cuántas veces aparece cada entero en `y`.
      Luego se divide por el número total de observaciones para obtener la frecuencia relativa.
    """
    a_priori = np.bincount(y.flatten().astype(int)) / y.size
    print(f"Probabilidad a priori de cada clase: {a_priori}")
    return np.log(a_priori)

  def _fit_params(self, X, y):
    """
    Método abstracto para estimar y almacenar los parámetros específicos del modelo 
    que definen la distribución condicional P(X | G=j) para cada clase j.

    Parámetros:
    -----------
    X : ndarray, forma (n_features, n_obs)
        Matriz de características, donde cada columna corresponde a una observación y 
        cada fila a una característica.
    y : ndarray, forma (1, n_obs)
        Vector de etiquetas de clase correspondiente a cada observación en X.

    Devuelve:
    ---------
    None
        Este método no devuelve valores; su propósito es calcular y guardar como atributos
        de la clase los parámetros que luego serán usados por `_predict_log_conditional`.
    """
    raise NotImplementedError()

  def _predict_log_conditional(self, x, class_idx):
    """
    Método abstracto para calcular el logaritmo de la probabilidad condicional P(x | G=class_idx).

    Parámetros:
    -----------
    x : ndarray, forma (n_features, 1)
        Vector de características de una observación.
    class_idx : int
        Índice de la clase para la que se quiere calcular la probabilidad.

    Devuelve:
    ---------
    float
        Logaritmo de P(x | G=class_idx).
    """
    raise NotImplementedError()

  def fit(self, X, y, a_priori=None):
    """
    Ajusta el clasificador a los datos de entrenamiento.

    Parámetros:
    -----------
    X : ndarray, forma (n_features, n_obs)
        Matriz de características (cada columna es una observación).
    y : ndarray, forma (1, n_obs)
        Vector de etiquetas de clase.
    a_priori : array-like o None
        Probabilidades a priori predefinidas para cada clase.
        Si es None, se estiman automáticamente a partir de y.

    Notas:
    ------
    - Si no se especifica a_priori, se calculan con `_estimate_a_priori`.
    - Luego se llama a `_fit_params` para estimar parámetros específicos del modelo.
    - `_fit_params` está al final para asegurar que las probabilidades a priori
      estén listas antes, por si el cálculo de parámetros las necesita.
    """
    self.log_a_priori = self._estimate_a_priori(y) if a_priori is None else np.log(a_priori)

    print(f"Log-Probabilidad a priori de cada clase: {self.log_a_priori}")

    self._fit_params(X, y)

  def predict(self, X):
    """
    Predice las clases para un conjunto de observaciones.

    Parámetros:
    -----------
    X : ndarray, forma (n_features, n_obs)
        Matriz de características.

    Devuelve:
    ---------
    y_hat : ndarray, forma (1, n_obs)
        Vector de predicciones de clase.
    """

    m_obs = X.shape[1]
    y_hat = np.empty(m_obs, dtype=int)

    for i in range(m_obs):
      y_hat[i] = self._predict_one(X[:,i].reshape(-1,1))

    return y_hat.reshape(1,-1)

  def _predict_one(self, x):
    """
    Predice la clase para una sola observación `x`.

    Parámetros:
    -----------
    x : ndarray, forma (n_features, 1)
        Vector de características.

    Devuelve:
    ---------
    int
        Índice de la clase con mayor probabilidad a posteriori.
    """

    log_posteriori = [ log_a_priori_i + self._predict_log_conditional(x, idx) for idx, log_a_priori_i
                  in enumerate(self.log_a_priori) ]

    return np.argmax(log_posteriori)
