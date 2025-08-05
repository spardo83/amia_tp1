import numpy as np
import numpy.linalg as LA
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg.lapack import dtrtri

class BaseBayesianClassifier:
    def __init__(self):
        pass

    def _estimate_a_priori(self, y):
        a_priori = np.bincount(y.flatten().astype(int)) / y.size
        return np.log(a_priori)

    def _fit_params(self, X, y):
        raise NotImplementedError()

    def _predict_log_conditional(self, x, class_idx):
        raise NotImplementedError()

    def fit(self, X, y, a_priori=None):
        self.log_a_priori = self._estimate_a_priori(y) if a_priori is None else np.log(a_priori)
        self._fit_params(X, y)

    def predict(self, X):
        m_obs = X.shape[1]
        y_hat = np.empty(m_obs, dtype=int)

        for i in range(m_obs):
            y_hat[i] = self._predict_one(X[:,i].reshape(-1,1))

        return y_hat.reshape(1,-1)

    def _predict_one(self, x):
        log_posteriori = [ log_a_priori_i + self._predict_log_conditional(x, idx) for idx, log_a_priori_i
                      in enumerate(self.log_a_priori) ]
        return np.argmax(log_posteriori)

class QDA(BaseBayesianClassifier):
    def _fit_params(self, X, y):
        self.inv_covs = [LA.inv(np.cov(X[:,y.flatten()==idx], bias=True))
                          for idx in range(len(self.log_a_priori))]
        self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                      for idx in range(len(self.log_a_priori))]

    def _predict_log_conditional(self, x, class_idx):
        inv_cov = self.inv_covs[class_idx]
        unbiased_x =  x - self.means[class_idx]
        return 0.5*np.log(LA.det(inv_cov)) -0.5 * unbiased_x.T @ inv_cov @ unbiased_x

class TensorizedQDA(QDA):
    def _fit_params(self, X, y):
        super()._fit_params(X,y)
        self.tensor_inv_cov = np.stack(self.inv_covs)
        self.tensor_means = np.stack(self.means)

    def _predict_log_conditionals(self,x):
        unbiased_x = x - self.tensor_means
        inner_prod = unbiased_x.transpose(0,2,1) @ self.tensor_inv_cov @ unbiased_x
        return 0.5*np.log(LA.det(self.tensor_inv_cov)) - 0.5 * inner_prod.flatten()

    def _predict_one(self, x):
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))

# Pregunta 3: Implementar FasterQDA (CORREGIDO)
class FasterQDA(TensorizedQDA):
    def predict(self, X):
        # Eliminar el ciclo for y predecir para todas las observaciones de una vez
        n_obs = X.shape[1]
        k_classes = len(self.log_a_priori)
        
        # Calcular log posteriores para todas las clases y observaciones
        log_posteriori = np.zeros((k_classes, n_obs))
        
        for class_idx in range(k_classes):
            # Para cada clase, calcular log posterior para todas las observaciones
            unbiased_x = X - self.tensor_means[class_idx:class_idx+1, :, :]
            
            # Calcular la forma cuadrática para cada observación
            for i in range(n_obs):
                x_i = unbiased_x[:, i:i+1]
                # Corregir: usar la matriz de covarianza correcta
                quadratic_form = x_i.T @ self.tensor_inv_cov[class_idx] @ x_i
                log_posteriori[class_idx, i] = (self.log_a_priori[class_idx] + 
                                               0.5*np.log(LA.det(self.tensor_inv_cov[class_idx])) - 
                                               0.5 * quadratic_form)
        
        # Retornar la clase con máximo log posterior para cada observación
        return np.argmax(log_posteriori, axis=0).reshape(1, -1)

# Pregunta 6: Implementar EfficientQDA (CORREGIDO)
class EfficientQDA(TensorizedQDA):
    def predict(self, X):
        # Usar la propiedad diag(A·B) = sum(A ⊙ B^T, axis=1)
        n_obs = X.shape[1]
        k_classes = len(self.log_a_priori)
        
        log_posteriori = np.zeros((k_classes, n_obs))
        
        for class_idx in range(k_classes):
            unbiased_x = X - self.tensor_means[class_idx:class_idx+1, :, :]  # Shape: (1, p, n)
            inv_cov = self.tensor_inv_cov[class_idx]  # Shape: (p, p)
            
            # Calcular A = unbiased_x.T (shape: n×p)
            A = unbiased_x[0].T  # Shape: (n, p) - tomar solo la primera dimensión
            
            # Calcular B = inv_cov @ unbiased_x (shape: p×n)
            B = inv_cov @ unbiased_x[0]  # Shape: (p, n) - usar unbiased_x[0]
            
            # Usar la propiedad: diag(A·B) = sum(A ⊙ B^T, axis=1)
            B_T = B.T  # Shape: (n, p)
            
            # A ⊙ B^T (element-wise multiplication)
            element_wise_prod = A * B_T  # Shape: (n, p)
            
            # Sumar a lo largo del axis=1 para obtener la diagonal
            quadratic_forms = np.sum(element_wise_prod, axis=1)  # Shape: (n,)
            
            # Calcular log posteriores
            log_posteriori[class_idx, :] = (self.log_a_priori[class_idx] + 
                                           0.5*np.log(LA.det(inv_cov)) - 
                                           0.5 * quadratic_forms)
        
        return np.argmax(log_posteriori, axis=0).reshape(1, -1)

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

# Pregunta 12: Implementar TensorizedChol
class TensorizedChol(QDA_Chol2):
    def _fit_params(self, X, y):
        super()._fit_params(X, y)
        # Stackear las matrices L y means
        self.tensor_Ls = np.stack(self.Ls)
        self.tensor_means = np.stack(self.means)
    
    def _predict_log_conditionals(self, x):
        unbiased_x = x - self.tensor_means  # Shape: (k, p, 1)
        
        # Resolver sistemas triangulares para todas las clases
        y_solutions = np.zeros((len(self.log_a_priori), x.shape[0], 1))
        
        for i in range(len(self.log_a_priori)):
            y_solutions[i] = solve_triangular(self.tensor_Ls[i], unbiased_x[i], lower=True)
        
        # Calcular log posteriores
        log_dets = -np.log(self.tensor_Ls.diagonal(axis1=1, axis2=2).prod(axis=1))
        quadratic_terms = -0.5 * (y_solutions**2).sum(axis=(1,2))
        
        return log_dets + quadratic_terms
    
    def _predict_one(self, x):
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))

# Pregunta 13: Implementar EfficientChol (CORREGIDO)
class EfficientChol(TensorizedChol):
    def predict(self, X):
        # Combinar insights de EfficientQDA y TensorizedChol
        n_obs = X.shape[1]
        k_classes = len(self.log_a_priori)
        
        log_posteriori = np.zeros((k_classes, n_obs))
        
        for class_idx in range(k_classes):
            unbiased_x = X - self.tensor_means[class_idx:class_idx+1, :, :]  # Shape: (1, p, n)
            L = self.tensor_Ls[class_idx]  # Shape: (p, p)
            
            # Resolver L*y = unbiased_x para todas las observaciones
            y_solutions = solve_triangular(L, unbiased_x[0], lower=True)  # Shape: (p, n) - usar unbiased_x[0]
            
            # Calcular ||y||^2 para cada observación (suma de cuadrados)
            quadratic_forms = np.sum(y_solutions**2, axis=0)  # Shape: (n,)
            
            # Calcular log posteriores
            log_det = -np.log(L.diagonal().prod())
            log_posteriori[class_idx, :] = (self.log_a_priori[class_idx] + 
                                           log_det - 0.5 * quadratic_forms)
        
        return np.argmax(log_posteriori, axis=0).reshape(1, -1) 