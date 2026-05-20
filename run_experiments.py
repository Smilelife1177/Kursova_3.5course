"""
run_experiments.py
==================
Порівняння трьох методів оптимізації на задачі XOR:
  1. Градієнтний спуск (фіксований крок)
  2. Метод найшвидшого спуску (точний line search)
  3. Метод Ньютона (чисельний Гессе + LM демпфування)

Виводить 3 графіки і зберігає їх у convergence_plots.png
"""

import sys
sys.path.insert(0, '/home/claude')   # новий optimizers.py
sys.path.append('/mnt/project')      # neural_network, xor_data

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from neural_network import MLP
from xor_data import X, y
from optimizers import Optimizers

# ─────────────────────────────────────────────
#  Запуск трьох методів
# ─────────────────────────────────────────────
print("=" * 60)
print("  ПОРІВНЯННЯ МЕТОДІВ ОПТИМІЗАЦІЇ  (XOR, MLP 2→4→1)")
print("=" * 60)

print("\n[1/3] Gradient Descent  (lr=1.0, max_iter=2000) ...")
net_gd = MLP(seed=42)
hist_gd, iters_gd, t_gd, lr = Optimizers.gradient_descent(
    net_gd, X, y, lr=1.0, max_iter=2000, tol=1e-6, verbose=False)
print(f"      → {iters_gd} ітерацій | {t_gd:.3f}с | loss={hist_gd[-1]:.2e}")

print("\n[2/3] Steepest Descent  (line search, max_iter=300) ...")
net_sd = MLP(seed=42)
hist_sd, iters_sd, t_sd = Optimizers.steepest_descent(
    net_sd, X, y, max_iter=300, tol=1e-6, verbose=False)
print(f"      → {iters_sd} ітерацій | {t_sd:.3f}с | loss={hist_sd[-1]:.2e}")

print("\n[3/3] Newton Method     (17×17 Hessian + LM, max_iter=50) ...")
net_nt = MLP(seed=42)
hist_nt, iters_nt, t_nt = Optimizers.newton_method(
    net_nt, X, y, max_iter=50, tol=1e-6, eps=1e-5, verbose=False)
print(f"      → {iters_nt} ітерацій | {t_nt:.3f}с | loss={hist_nt[-1]:.2e}")

# ─────────────────────────────────────────────
#  Підготовка даних для графіків
# ─────────────────────────────────────────────
colors = {
    'gd':  '#E74C3C',   # червоний
    'sd':  '#2ECC71',   # зелений
    'nt':  '#3498DB',   # синій
}

labels = {
    'gd': f'Градієнтний спуск (lr={lr})',
    'sd': 'Метод найшвидшого спуску',
    'nt': 'Метод Ньютона',
}

# ─────────────────────────────────────────────
#  Малюємо 3 графіки на одному аркуші
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 13))
fig.patch.set_facecolor('#F8F9FA')

gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.38, wspace=0.30,
                       left=0.07, right=0.97,
                       top=0.91, bottom=0.07)

ax1 = fig.add_subplot(gs[0, :])   # верхній широкий
ax2 = fig.add_subplot(gs[1, 0])   # нижній лівий
ax3 = fig.add_subplot(gs[1, 1])   # нижній правий

# ── Стиль осей ──────────────────────────────
for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('#FFFFFF')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.4, color='#AAAAAA')

# ════════════════════════════════════════════
#  Графік 1: Збіжність у лог-масштабі (всі методи)
# ════════════════════════════════════════════
ax1.plot(hist_gd, color=colors['gd'], lw=2.2, label=labels['gd'])
ax1.plot(hist_sd, color=colors['sd'], lw=2.2, label=labels['sd'])
ax1.plot(hist_nt, color=colors['nt'], lw=2.5, label=labels['nt'],
         marker='o', markersize=4, markevery=3)
ax1.set_yscale('log')
ax1.set_xlabel('Ітерація', fontsize=12)
ax1.set_ylabel('Функція втрат L (log)', fontsize=12)
ax1.set_title('Збіжність методів оптимізації на задачі XOR  (MSE, лог-масштаб)',
              fontsize=13, fontweight='bold', pad=10)
ax1.legend(fontsize=11, framealpha=0.9)

# Анотації фінальних значень
for hist, col in [(hist_gd, colors['gd']),
                  (hist_sd, colors['sd']),
                  (hist_nt, colors['nt'])]:
    ax1.annotate(f'{hist[-1]:.1e}',
                 xy=(len(hist)-1, hist[-1]),
                 xytext=(8, 0), textcoords='offset points',
                 fontsize=9, color=col, va='center')

# ════════════════════════════════════════════
#  Графік 2: Ітерації до збіжності (bar chart)
# ════════════════════════════════════════════
methods  = ['GD', 'Steepest\nDescent', 'Newton']
iters_   = [iters_gd, iters_sd, iters_nt]
bar_cols = [colors['gd'], colors['sd'], colors['nt']]

bars = ax2.bar(methods, iters_, color=bar_cols, width=0.5,
               edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, iters_):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.set_ylabel('Кількість ітерацій', fontsize=12)
ax2.set_title('Ітерацій до зупинки', fontsize=12, fontweight='bold')
ax2.set_ylim(0, max(iters_) * 1.18)

# ════════════════════════════════════════════
#  Графік 3: Час vs фінальний Loss (scatter)
# ════════════════════════════════════════════
times_  = [t_gd,       t_sd,       t_nt]
losses_ = [hist_gd[-1], hist_sd[-1], hist_nt[-1]]

for t, l, col, lbl in zip(times_, losses_, bar_cols, ['GD', 'SD', 'NT']):
    ax3.scatter(t, l, color=col, s=180, zorder=5, edgecolors='white', linewidth=1.5)
    ax3.annotate(lbl, xy=(t, l), xytext=(8, 4),
                 textcoords='offset points', fontsize=11,
                 color=col, fontweight='bold')

ax3.set_yscale('log')
ax3.set_xlabel('Час виконання (с)', fontsize=12)
ax3.set_ylabel('Фінальний Loss (log)', fontsize=12)
ax3.set_title('Час виконання vs Якість результату', fontsize=12, fontweight='bold')

# ── Загальний заголовок ──────────────────────
fig.suptitle(
    'Градієнтні методи оптимізації нейромережі  |  Архітектура: MLP 2→4→1  |  Задача: XOR',
    fontsize=14, fontweight='bold', y=0.97, color='#2C3E50')

# ─────────────────────────────────────────────
#  Таблиця в консолі
# ─────────────────────────────────────────────
print("\n" + "=" * 62)
print(f"  {'Метод':<24} {'Ітерацій':>9} {'Час (с)':>8} {'Loss':>13}")
print("  " + "─" * 58)
rows = [
    ('Gradient Descent',  iters_gd, t_gd, hist_gd[-1]),
    ('Steepest Descent',  iters_sd, t_sd, hist_sd[-1]),
    ('Newton Method',     iters_nt, t_nt, hist_nt[-1]),
]
for name, it, t, l in rows:
    print(f"  {name:<24} {it:>9} {t:>8.3f} {l:>13.2e}")
print("=" * 62)

# ─────────────────────────────────────────────
#  Перевірка передбачень після тренування
# ─────────────────────────────────────────────
print("\n  Передбачення найкращої моделі (Newton):")
print("  " + "─" * 30)
preds = net_nt.forward(X)
for xi, yi, pi in zip(X, y.ravel(), preds.ravel()):
    ok = "✓" if abs(pi - yi) < 0.1 else "✗"
    print(f"  {int(xi[0])} XOR {int(xi[1])} = {yi}  →  ŷ={pi:.4f}  {ok}")

# ─────────────────────────────────────────────
#  Зберігаємо графік
# ─────────────────────────────────────────────
out_path = 'convergence_plots.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\n  Графік збережено: {out_path}")