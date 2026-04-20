from neural_network import MLP
from xor_data import X, y
from optimizers import Optimizers
import numpy as np

# Тест базової мережі
net = MLP(seed=42)
print("Initial loss:", net.loss(net.forward(X), y))

# Тест GD
net = MLP(seed=42)  # скидаємо
history_gd, iters, time_gd = Optimizers.gradient_descent(net, X, y, lr=0.5, max_iter=1000, verbose=True)

print(f"\nGD finished in {iters} iterations, time: {time_gd:.3f}s")
print("Final loss:", history_gd[-1])