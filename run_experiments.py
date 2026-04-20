from neural_network import MLP
from xor_data import X, y
from optimizers import Optimizers
import matplotlib.pyplot as plt
import numpy as np

print("=== Порівняння методів оптимізації для XOR ===")

# 1. Градієнтний спуск
print("\nЗапуск Gradient Descent (lr=1.0)...")
net_gd = MLP(seed=42)
hist_gd, iters_gd, time_gd, lr = Optimizers.gradient_descent(net_gd, X, y, lr=1.0, verbose=True)
print(f"GD → {iters_gd} ітерацій, час: {time_gd:.3f}с, фінальний loss: {hist_gd[-1]:.6f}")

# 2. Метод найшвидшого спуску
print("\nЗапуск Steepest Descent...")
net_sd = MLP(seed=42)
hist_sd, iters_sd, time_sd = Optimizers.steepest_descent(net_sd, X, y, verbose=True)
print(f"Steepest Descent → {iters_sd} ітерацій, час: {time_sd:.3f}с, фінальний loss: {hist_sd[-1]:.6f}")

# 3. Метод Ньютона (тимчасова версія)
print("\nЗапуск Newton Method...")
net_newton = MLP(seed=42)
hist_newton, iters_newton, time_newton = Optimizers.newton_method(net_newton, X, y, verbose=True)
print(f"Newton → {iters_newton} ітерацій, час: {time_newton:.3f}с, фінальний loss: {hist_newton[-1]:.6f}")

# Побудова графіка (збережеться як convergence.png)
plt.figure(figsize=(10, 6))
plt.plot(hist_gd, label=f'Gradient Descent (lr={lr})', linewidth=2)
plt.plot(hist_sd, label='Steepest Descent', linewidth=2)
plt.plot(hist_newton, label='Newton Method', linewidth=2)
plt.yscale('log')
plt.xlabel('Ітерація')
plt.ylabel('Функція втрат (log scale)')
plt.title('Збіжність методів оптимізації на задачі XOR')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('convergence.png', dpi=300, bbox_inches='tight')
plt.show()

# Таблиця результатів
print("\n=== Порівняльна таблиця ===")
print(f"{'Метод':<20} {'Ітерацій':<10} {'Час (с)':<10} {'Фін. loss':<12}")
print("-" * 52)
print(f"{'Gradient Descent':<20} {iters_gd:<10} {time_gd:<10.3f} {hist_gd[-1]:<12.6f}")
print(f"{'Steepest Descent':<20} {iters_sd:<10} {time_sd:<10.3f} {hist_sd[-1]:<12.6f}")
print(f"{'Newton Method':<20} {iters_newton:<10} {time_newton:<10.3f} {hist_newton[-1]:<12.6f}")