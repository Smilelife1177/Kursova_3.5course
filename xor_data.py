import numpy as np

# Датасет XOR: 4 приклади, 2 ознаки
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)

y = np.array([
    [0],  # 0 XOR 0 = 0
    [1],  # 0 XOR 1 = 1
    [1],  # 1 XOR 0 = 1
    [0],  # 1 XOR 1 = 0
], dtype=float)