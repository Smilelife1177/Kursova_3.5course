import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ==================== 1. Contour plot + траєкторії ====================
def plot_optimization_paths():
    def loss(w):
        x, y = w
        return 0.1*(x**2) + 2*(y**2) + 0.5*np.sin(3*x) + 0.3*np.cos(4*y)

    def grad(w):
        x, y = w
        gx = 0.2*x + 1.5*np.cos(3*x)
        gy = 4*y - 1.2*np.sin(4*y)
        return np.array([gx, gy])

    # Метод найшвидшого спуску (з Line Search)
    def steepest_descent(w0, max_iter=12):
        path = [w0.copy()]
        w = w0.copy()
        for i in range(max_iter):
            g = grad(w)
            if np.linalg.norm(g) < 1e-6:
                break

            def line_search_eta(eta):
                return loss(w - eta * g)

            # Виправлення: використовуємо bracket замість bounds
            res = minimize_scalar(line_search_eta, 
                                  bracket=(0, 0.01, 2.0),   # початкові точки для пошуку
                                  method='brent',
                                  tol=1e-6)
            eta_opt = res.x
            w = w - eta_opt * g
            path.append(w.copy())
            
            print(f"Ітерація {i+1:2d} | η = {eta_opt:.4f} | Loss = {loss(w):.6f}")
        return np.array(path)

    # Звичайний Gradient Descent
    def gradient_descent(w0, eta=0.08, max_iter=25):
        path = [w0.copy()]
        w = w0.copy()
        for _ in range(max_iter):
            g = grad(w)
            w = w - eta * g
            path.append(w.copy())
        return np.array(path)

    # Створення контурного графіка
    x = np.linspace(-3.5, 3.5, 250)
    y = np.linspace(-3.5, 3.5, 250)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[loss([xi, yi]) for xi in x] for yi in y])

    plt.figure(figsize=(12, 8))
    plt.contour(X, Y, Z, levels=50, cmap='viridis', alpha=0.75)
    plt.contour(X, Y, Z, levels=15, colors='black', linewidths=0.5, alpha=0.4)

    w0 = np.array([-2.8, 2.6])

    path_gd = gradient_descent(w0, eta=0.08, max_iter=22)
    path_sd = steepest_descent(w0, max_iter=12)

    plt.plot(path_gd[:,0], path_gd[:,1], 'o-', color='#e74c3c', 
             label='Gradient Descent (фіксований η=0.08)', linewidth=2, markersize=4)
    
    plt.plot(path_sd[:,0], path_sd[:,1], 'o-', color='#2ecc71', 
             label='Steepest Descent (Line Search + Brent)', linewidth=2.5, markersize=5)

    plt.xlabel('$w_1$', fontsize=14)
    plt.ylabel('$w_2$', fontsize=14)
    plt.title('Порівняння траєкторій оптимізації\nGradient Descent vs Метод найшвидшого спуску', 
              fontsize=15, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("Steepest_Descent_vs_GD.png", dpi=300, bbox_inches='tight')
    print("\n✅ Графік збережено: Steepest_Descent_vs_GD.png")
    plt.show()


# ==================== 2. Ілюстрація Line Search ====================
def plot_line_search():
    eta_values = np.linspace(0, 4, 300)
    
    # Приклад функції вздовж напрямку спуску
    def loss_along_line(eta):
        return (eta - 2.5)**2 + 0.5*np.sin(8*eta) + 1.0   # з невеликою осциляцією

    loss_values = [loss_along_line(eta) for eta in eta_values]

    plt.figure(figsize=(10, 6))
    plt.plot(eta_values, loss_values, 'b-', linewidth=2.5, label=r'$L(\mathbf{w} - \eta \nabla L(\mathbf{w}))$')
    
    plt.axvline(x=2.5, color='red', linestyle='--', linewidth=2, label=r'Оптимальне $\eta \approx 2.5$')
    plt.scatter([2.5], [loss_along_line(2.5)], color='red', s=120, zorder=5, edgecolors='black')

    plt.title('Ілюстрація Line Search в методі найшвидшого спуску', fontsize=14)
    plt.xlabel('Розмір кроку η', fontsize=13)
    plt.ylabel('Значення функції втрат', fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.savefig("Line_Search_Illustration.png", dpi=300, bbox_inches='tight')
    print("✅ Графік збережено: Line_Search_Illustration.png")
    plt.show()


if __name__ == "__main__":
    plot_optimization_paths()
    plot_line_search()