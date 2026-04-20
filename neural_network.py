import numpy as np

class MLP:
    """
    Багатошаровий перцептрон (MLP) для задачі XOR.
    Архітектура: 2 -> 4 -> 1, активація сигмоїда.
    """

    def __init__(self, seed=42):
        np.random.seed(seed)
        # Ваги: W1 (2x4), b1 (4,), W2 (4x1), b2 (1,)
        self.W1 = np.random.randn(2, 4) * 0.5
        self.b1 = np.zeros(4)
        self.W2 = np.random.randn(4, 1) * 0.5
        self.b2 = np.zeros(1)

    def sigmoid(self, z):
        """Сигмоїдна функція активації: σ(z) = 1 / (1 + e^(-z))"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_deriv(self, a):
        """Похідна сигмоїди: σ'(z) = σ(z)·(1 - σ(z))"""
        return a * (1.0 - a)

    def forward(self, X):
        """
        Прямий прохід (Forward Pass).
        X: (n_samples, 2) -> повертає y_hat: (n_samples, 1)
        """
        self.X  = X
        self.Z1 = X @ self.W1 + self.b1        # (n, 4)
        self.A1 = self.sigmoid(self.Z1)         # (n, 4)
        self.Z2 = self.A1 @ self.W2 + self.b2  # (n, 1)
        self.A2 = self.sigmoid(self.Z2)         # (n, 1)
        return self.A2

    def loss(self, y_hat, y):
        """MSE функція втрат: L = (1/n) * Σ(y - ŷ)²"""
        return np.mean((y - y_hat) ** 2)

    def backward(self, y_hat, y):
        """
        Зворотне поширення помилки (Backpropagation).
        Повертає словник градієнтів: dW1, db1, dW2, db2
        """
        n = y.shape[0]

        # Похідна втрат по A2
        dL_dA2 = -2 / n * (y - y_hat)              # (n, 1)
        dA2_dZ2 = self.sigmoid_deriv(self.A2)      # (n, 1)
        dZ2 = dL_dA2 * dA2_dZ2                     # (n, 1)

        dW2 = self.A1.T @ dZ2                      # (4, 1)
        db2 = dZ2.sum(axis=0)                     # (1,)

        # Зворотне поширення до прихованого шару
        dA1 = dZ2 @ self.W2.T                      # (n, 4)
        dZ1 = dA1 * self.sigmoid_deriv(self.A1)   # (n, 4)

        dW1 = self.X.T @ dZ1                       # (2, 4)
        db1 = dZ1.sum(axis=0)                     # (4,)

        return {'dW1': dW1, 'db1': db1,
                'dW2': dW2, 'db2': db2}

    def get_params(self):
        """Повертає всі параметри як один вектор (для методу Ньютона)"""
        return np.concatenate([
            self.W1.ravel(), self.b1,
            self.W2.ravel(), self.b2
        ])

    def set_params(self, params):
        """Встановлює параметри з вектора (для методу Ньютона)"""
        i = 0
        self.W1 = params[i:i+8].reshape(2, 4);  i += 8
        self.b1 = params[i:i+4];              i += 4
        self.W2 = params[i:i+4].reshape(4, 1); i += 4
        self.b2 = params[i:i+1]