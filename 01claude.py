import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from neural_network import MLP
from xor_data import X, y
from optimizers import Optimizers

class OptimizerDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Gradient Optimizer Demo")
        
        # Стан
        self.net = None
        self.history = []
        self.method = tk.StringVar(value="GD")
        self.lr_var = tk.DoubleVar(value=1.0)
        
        self._build_ui()
        self.reset()

    def _build_ui(self):
        # Ліва панель керування
        ctrl = tk.Frame(self.root, width=200)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        tk.Label(ctrl, text="Метод:").pack(anchor="w")
        for m in ["GD", "Steepest", "Newton"]:
            tk.Radiobutton(ctrl, text=m, variable=self.method,
                           value=m, command=self.reset).pack(anchor="w")
        
        tk.Label(ctrl, text="Learning rate (GD):").pack(anchor="w", pady=(10,0))
        tk.Scale(ctrl, from_=0.1, to=3.0, resolution=0.1,
                 variable=self.lr_var, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        tk.Button(ctrl, text="Скинути", command=self.reset).pack(fill=tk.X, pady=5)
        tk.Button(ctrl, text="Крок →", command=self.step).pack(fill=tk.X)
        tk.Button(ctrl, text="▶ До збіжності", command=self.run_all).pack(fill=tk.X, pady=5)
        
        self.iter_label = tk.Label(ctrl, text="Ітерація: 0")
        self.iter_label.pack()
        self.loss_label = tk.Label(ctrl, text="Loss: —")
        self.loss_label.pack()
        
        # Matplotlib canvas
        self.fig, (self.ax_loss, self.ax_pred) = plt.subplots(1, 2, figsize=(9, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def reset(self):
        self.net = MLP(seed=42)
        self.history = []
        self._update_plot()

    def step(self):
        """Виконати один крок поточного оптимізатора"""
        y_hat = self.net.forward(X)
        loss = self.net.loss(y_hat, y)
        self.history.append(loss)
        
        grads = self.net.backward(y_hat, y)
        
        if self.method.get() == "GD":
            lr = self.lr_var.get()
            self.net.W1 -= lr * grads['dW1']
            self.net.b1 -= lr * grads['db1']
            self.net.W2 -= lr * grads['dW2']
            self.net.b2 -= lr * grads['db2']
        elif self.method.get() == "Steepest":
            # один крок steepest descent
            ...  # виклик line search
        else:
            # один крок Newton
            ...  # виклик з Гессіаном
        
        self._update_plot()

    def run_all(self):
        for _ in range(5000):
            self.step()
            if self.history and self.history[-1] < 1e-6:
                break
        self._update_plot()

    def _update_plot(self):
        # Крива збіжності
        self.ax_loss.clear()
        if self.history:
            self.ax_loss.semilogy(self.history)
        self.ax_loss.set_title("Loss (log)")
        self.ax_loss.set_xlabel("Ітерація")
        self.ax_loss.grid(True, alpha=0.3)
        
        # Поточні передбачення
        self.ax_pred.clear()
        preds = self.net.forward(X).flatten()
        labels = ["0⊕0", "0⊕1", "1⊕0", "1⊕1"]
        colors = ["green" if abs(p - t) < 0.1 else "red"
                  for p, t in zip(preds, [0,1,1,0])]
        bars = self.ax_pred.bar(labels, preds, color=colors)
        self.ax_pred.axhline(0.5, color='gray', linestyle='--')
        self.ax_pred.set_ylim(0, 1)
        self.ax_pred.set_title("Передбачення мережі")
        
        n = len(self.history)
        loss_str = f"{self.history[-1]:.2e}" if self.history else "—"
        self.iter_label.config(text=f"Ітерація: {n}")
        self.loss_label.config(text=f"Loss: {loss_str}")
        
        self.fig.tight_layout()
        self.canvas.draw()

root = tk.Tk()
app = OptimizerDemo(root)
root.mainloop()