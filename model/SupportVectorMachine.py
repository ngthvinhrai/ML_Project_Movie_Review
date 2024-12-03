import numpy as np
from cvxopt import matrix, solvers

class SVM:
  def __init__(self, kernel, C=0, degree=3, coef=0.0, gamma = 'scale', epsilon=1e-6):
    self.kernel = kernel
    self.C = C
    self.degree = degree
    self.coef = coef
    self.gamma = gamma
    self.epsilon = epsilon

  def _linear_kernel(self, X, Z):
    return np.dot(X,Z.T)

  def _polynomial_kernel(self, X, Z):
    return (np.dot(X,Z.T) + self.coef)**self.degree

  def _gaussianRBF(self, X, Z):
    K = np.zeros((X.shape[0], X.shape[0]))

    for i in range(X.shape[0]):
      for j in range(i,X.shape[0]):
        K[i,j] = np.exp(-np.linalg.norm(X[i], Z[j])**2/(2*self.sigma**2))
        K[j,i] = K[i,j]

    return K

  def _sigmoid(self, X, Z):
    return np.tanh(self.sigma*np.dot(X, Z.T) + self.coef)

  def fit(self, X, y):
    assert y.shape == (X.shape[0], 1)

    match self.gamma:
      case 'scale':
        self.sigma = X.shape[0]*X.var()
      case 'auto':
        self.sigma = X.shape[0]

    match self.kernel:
      case 'linear':
        self.kernel_matrix = self._linear_kernel
      case 'polynomial':
        self.kernel_matrix = self._polynomial_kernel
      case 'rbf':
        self.kernel_matrix = self._gaussianRBF
      case 'sigmoid':
        self.kernel_matrix = self._sigmoid

    K = matrix(y.dot(y.T)*self.kernel_matrix(X, X).astype(np.float64))
    q = matrix(-np.ones((X.shape[0], 1)))

    if self.C == 0:
      G = matrix(-np.identity(X.shape[0]))
      h = matrix(np.zeros((X.shape[0], 1)))
    else:
      G1 = -np.identity(X.shape[0])
      G2 = np.identity(X.shape[0])
      G = matrix(np.vstack((G1, G2)))
      h1 = np.zeros((X.shape[0], 1))
      h2 = np.zeros((X.shape[0], 1)) + self.C
      h = matrix(np.vstack((h1,h2)))

    A = matrix(y.T.astype(np.float64))
    b = matrix(np.zeros((1,1)))

    sol = solvers.qp(K, q, G, h, A, b)
    self.lamda = np.array(sol['x'])

    lamda_idx = np.where(self.lamda > self.epsilon)[0]
    self.support_lamda = self.lamda[lamda_idx]
    self.support_vector = X[lamda_idx]
    self.support_output = y[lamda_idx]

    self.b = np.mean(self.support_output - self.kernel_matrix(self.support_vector, self.support_output*self.support_vector).dot(self.support_lamda))
    if self.kernel == 'linear': self.W = (self.support_output*self.support_vector).T.dot(self.support_lamda)

    return self

  def predict(self, X):
    return np.sign(self.kernel_matrix(X, self.support_output*self.support_vector).dot(self.support_lamda) + self.b)


if __name__ == "__main__":
    pass