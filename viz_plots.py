import numpy as np
import matplotlib.pyplot as plt

# 1. Визначення цільової функції (Еліптична парабола)
# f(x, y) = 0.5 * (x^2 + 10 * y^2)
# Це класична функція для демонстрації різниці між методами
def f(x, y):
    return 0.5 * (x**2 + 10 * y**2)

def grad_f(x, y):
    return np.array([x, 10 * y])

def hessian_f(x, y):
    return np.array([[1, 0], [0, 10]])

# Налаштування сітки для графіків
x_range = np.linspace(-10, 10, 100)
y_range = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

def plot_method(path, title, filename, color):
    # 1. Створюємо малюнок та осі
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 2. Встановлюємо темно-синій фон для всього малюнка та робочої зони
    bg_color = '#0d1b2a'
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # 3. Малюємо лінії рівня (використовуємо іншу палітру 'plasma' або 'cool' для темного фону)
    plt.contour(X, Y, Z, levels=20, cmap='plasma', alpha=0.6)
    
    # 4. Будуємо шлях методу оптимізації
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], '-o', color=color, markersize=4, label=title)
    
    # 5. Нове позначення мінімуму (білий трикутник замість червоної зірки)
    plt.plot(0, 0, 'w^', markersize=8, label='Мінімум')
    
    # 6. Фарбуємо заголовки та підписи осей у білий колір
    plt.title(title, color='white', fontsize=12, pad=15)
    plt.xlabel('Параметр w1', color='white')
    plt.ylabel('Параметр w2', color='white')
    
    # 7. Налаштовуємо рамку графіка та мітки осей (шкалу)
    ax.spines['bottom'].set_color('#415a77') # світліша рамка
    ax.spines['top'].set_color('#415a77')
    ax.spines['left'].set_color('#415a77')
    ax.spines['right'].set_color('#415a77')
    ax.tick_params(colors='white') # білі цифри на осях
    
    # 8. Налаштовуємо легенду (прозорий або темний фон, білий текст)
    legend = plt.legend(facecolor='#1b263b', edgecolor='#415a77')
    plt.setp(legend.get_texts(), color='white')
    
    # 9. Тонка світла сітка
    plt.grid(color='white', alpha=0.1)
    
    # Зберігаємо графік із правильним кольором фону (facecolor=fig.get_facecolor())
    plt.savefig(filename, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Графік збережено: {filename}")
    plt.close()


# --- 1. Gradient Descent (GD) ---
def run_gd(start_pos, lr=0.1, steps=20):
    path = [start_pos]
    curr = np.array(start_pos, dtype=float)
    for _ in range(steps):
        g = grad_f(curr[0], curr[1])
        curr -= lr * g
        path.append(curr.copy())
    return path

# --- 2. Steepest Descent (SD) ---
def run_sd(start_pos, steps=10):
    path = [start_pos]
    curr = np.array(start_pos, dtype=float)
    for _ in range(steps):
        g = grad_f(curr[0], curr[1])
        # Оптимальний крок для квадратичної форми: alpha = (g'g) / (g'Hg)
        H = hessian_f(curr[0], curr[1])
        alpha = np.dot(g, g) / np.dot(g, np.dot(H, g))
        curr -= alpha * g
        path.append(curr.copy())
    return path

# --- 3. Newton's Method ---
def run_newton(start_pos, steps=5):
    path = [start_pos]
    curr = np.array(start_pos, dtype=float)
    for _ in range(steps):
        g = grad_f(curr[0], curr[1])
        H = hessian_f(curr[0], curr[1])
        # крок Ньютона: x = x - H^-1 * g
        step = np.linalg.solve(H, g)
        curr -= step
        path.append(curr.copy())
    return path

# Запуск та генерація
start = [9.0, 4.0]

path_gd = run_gd(start, lr=0.15, steps=30)
plot_method(path_gd, "Gradient Descent (Фіксований крок)", "plot_gd.png", "green")

path_sd = run_sd(start, steps=15)
plot_method(path_sd, "Steepest Descent (Оптимальний крок)", "plot_sd.png", "orange")

path_newton = run_newton(start, steps=5)
plot_method(path_newton, "Newton's Method (Друга похідна)", "plot_newton.png", "purple")

print("\nУспішно згенеровано 3 порівняльні графіки.")
