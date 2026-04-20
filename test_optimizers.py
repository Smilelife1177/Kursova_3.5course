import sys
sys.path.insert(0, '/home/claude')   # наш новий optimizers.py
sys.path.append('/mnt/project')      # neural_network та xor_data

from neural_network import MLP
from xor_data import X, y
from optimizers import Optimizers
import numpy as np

print("=" * 55)
print("  ТЕСТ МЕТОДІВ ОПТИМІЗАЦІЇ (XOR)")
print("=" * 55)

# ── 1. Градієнтний спуск ──────────────────────────────────
print("\n[1] Gradient Descent (lr=1.0, max_iter=2000)")
net = MLP(seed=42)
hist_gd, iters_gd, t_gd, lr = Optimizers.gradient_descent(
    net, X, y, lr=1.0, max_iter=2000, tol=1e-6, verbose=True)
print(f"    → {iters_gd} ітерацій | час: {t_gd:.3f}с | loss: {hist_gd[-1]:.8f}")

# ── 2. Метод найшвидшого спуску ───────────────────────────
print("\n[2] Steepest Descent (точний line search)")
net = MLP(seed=42)
hist_sd, iters_sd, t_sd = Optimizers.steepest_descent(
    net, X, y, max_iter=200, tol=1e-6, verbose=True)
print(f"    → {iters_sd} ітерацій | час: {t_sd:.3f}с | loss: {hist_sd[-1]:.8f}")

# ── 3. Метод Ньютона ──────────────────────────────────────
print("\n[3] Newton Method (17×17 Hessian + LM damping)")
net = MLP(seed=42)
hist_nt, iters_nt, t_nt = Optimizers.newton_method(
    net, X, y, max_iter=50, tol=1e-6, eps=1e-5, verbose=True)
print(f"    → {iters_nt} ітерацій | час: {t_nt:.3f}с | loss: {hist_nt[-1]:.8f}")

# ── Таблиця порівняння ────────────────────────────────────
print("\n" + "=" * 55)
print(f"  {'Метод':<22} {'Ітерацій':>9} {'Час (с)':>8} {'Loss':>12}")
print("  " + "-" * 53)
print(f"  {'Gradient Descent':<22} {iters_gd:>9} {t_gd:>8.3f} {hist_gd[-1]:>12.2e}")
print(f"  {'Steepest Descent':<22} {iters_sd:>9} {t_sd:>8.3f} {hist_sd[-1]:>12.2e}")
print(f"  {'Newton Method':<22} {iters_nt:>9} {t_nt:>8.3f} {hist_nt[-1]:>12.2e}")
print("=" * 55)