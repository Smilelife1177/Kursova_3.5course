import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import os

def create_optimization_gui():
    root = tk.Tk()
    root.title("Порівняльна характеристика методів оптимізації")
    root.geometry("1250x720")
    root.configure(bg="#f0f0f0")

    # ==================== Заголовок ====================
    title = tk.Label(root, text="Порівняльна характеристика методів оптимізації",
                     font=tkfont.Font(family="Helvetica", size=18, weight="bold"),
                     bg="#f0f0f0", fg="#1a3c6e")
    title.pack(pady=10)

    # ==================== Таблиця ====================
    frame_table = tk.Frame(root, bg="#f0f0f0")
    frame_table.pack(pady=5, padx=20, fill="x")

    columns = ("char", "gd", "sd", "newton")
    tree = ttk.Treeview(frame_table, columns=columns, show="headings", height=7)

    tree.heading("char", text="Характеристика")
    tree.heading("gd", text="Градієнтний спуск")
    tree.heading("sd", text="Метод найшвидшого спуску")
    tree.heading("newton", text="Метод Ньютона")

    tree.column("char", width=280, anchor="w")
    tree.column("gd", width=220, anchor="center")
    tree.column("sd", width=220, anchor="center")
    tree.column("newton", width=220, anchor="center")

    data = [
        ("Швидкість збіжності", "Лінійна", "Суперіорна лінійна", "Квадратична"),
        ("Обчислення на ітерацію", "Низьке", "Середнє/високе", "Високе"),
        ("Потреба в налаштуванні η", "Так", "Ні", "Ні"),
        ("Пам’ять", "O(d)", "O(d)", "O(d²)"),
        ("Стабільність", "Висока", "Висока", "Середня"),
        ("Придатність для великих мереж", "Відмінна", "Добра", "Погана"),
    ]

    for i, row in enumerate(data):
        tree.insert("", "end", values=row)

    tree.pack(side="left", fill="both", expand=True)

    info = tk.Label(root, text="d — кількість параметрів (у нашій задачі d = 17)",
                    font=("Helvetica", 10, "italic"), bg="#f0f0f0", fg="#444")
    info.pack(pady=5)

    # ==================== Графіки ====================
    fig = Figure(figsize=(11, 5), dpi=100)
    axs = fig.subplots(1, 3, sharey=True)

    methods = ["Gradient Descent", "Steepest Descent", "Newton's Method"]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    convergence = [0.85, 0.65, 0.25]   # умовна швидкість збіжності

    for i, ax in enumerate(axs):
        # Симуляція збіжності
        iterations = np.arange(0, 31)
        error = np.exp(-convergence[i] * iterations) + np.random.normal(0, 0.015, 31)
        error = np.maximum(error, 1e-6)
        
        ax.plot(iterations, error, 'o-', color=colors[i], linewidth=2.5, markersize=4)
        ax.set_title(methods[i], fontsize=11, fontweight='bold')
        ax.set_xlabel("Ітерації")
        ax.set_ylabel("Логарифмічна похибка" if i == 0 else "")
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, root)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10, padx=20, fill="both", expand=True)

    # ==================== Кнопки збереження ====================
    btn_frame = tk.Frame(root, bg="#f0f0f0")
    btn_frame.pack(pady=10)

    def save_table():
        # Збереження таблиці як зображення (matplotlib)
        fig_table = plt.figure(figsize=(10, 4))
        ax = fig_table.add_subplot(111)
        ax.axis('off')
        
        table_data = [["Характеристика", "Градієнтний спуск", "Найшвидший спуск", "Метод Ньютона"]] + data
        table = ax.table(cellText=table_data, loc='center', cellLoc='center', bbox=[0,0,1,1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        plt.savefig("Порівняльна_таблиця.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Таблиця збережена як 'Порівняльна_таблиця.png'")

    def save_plots():
        fig.savefig("Порівняння_збіжності.png", dpi=300, bbox_inches='tight')
        print("✅ Графіки збережена як 'Порівняння_збіжності.png'")

    tk.Button(btn_frame, text="💾 Зберегти таблицю", command=save_table,
              bg="#2980b9", fg="white", font=("Helvetica", 10, "bold"), padx=15, pady=8).pack(side="left", padx=10)

    tk.Button(btn_frame, text="💾 Зберегти графіки", command=save_plots,
              bg="#27ae60", fg="white", font=("Helvetica", 10, "bold"), padx=15, pady=8).pack(side="left", padx=10)

    tk.Button(btn_frame, text="Закрити", command=root.destroy,
              bg="#c0392b", fg="white", font=("Helvetica", 10, "bold"), padx=15, pady=8).pack(side="left", padx=10)

    root.mainloop()


if __name__ == "__main__":
    create_optimization_gui()