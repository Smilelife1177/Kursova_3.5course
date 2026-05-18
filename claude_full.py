"""
Gradient Optimization Methods Visualizer
Курсова робота: Градієнтні методи оптимізації для навчання нейромережі
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import time

# ─── Palette ───────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
BORDER   = "#30363d"
ACCENT1  = "#58a6ff"   # blue  – GD
ACCENT2  = "#3fb950"   # green – Steepest
ACCENT3  = "#f78166"   # red   – Newton
ACCENT_Y = "#e3b341"   # gold  – highlight
FG       = "#e6edf3"
FG_DIM   = "#8b949e"

# ─── Neural Network ────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1 - s)

X_XOR = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
Y_XOR = np.array([[0],[1],[1],[0]], dtype=float)

class MLP:
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (4, 2))
        self.b1 = np.zeros((4, 1))
        self.W2 = rng.normal(0, 0.5, (1, 4))
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        self.X  = X.T
        self.Z1 = self.W1 @ self.X + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def loss(self, X, y):
        yhat = self.forward(X)
        return float(np.mean((y.T - yhat) ** 2))

    def backward(self, y):
        n  = y.shape[0]
        dZ2 = (self.A2 - y.T) * sigmoid_d(self.Z2) * (2/n)
        dW2 = dZ2 @ self.A1.T
        db2 = np.sum(dZ2, axis=1, keepdims=True)
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * sigmoid_d(self.Z1)
        dW1 = dZ1 @ self.X.T
        db1 = np.sum(dZ1, axis=1, keepdims=True)
        return dW1, db1, dW2, db2

    def get_params(self):
        return np.concatenate([self.W1.ravel(), self.b1.ravel(),
                                self.W2.ravel(), self.b2.ravel()])

    def set_params(self, p):
        i = 0
        self.W1 = p[i:i+8].reshape(4,2);  i += 8
        self.b1 = p[i:i+4].reshape(4,1);  i += 4
        self.W2 = p[i:i+4].reshape(1,4);  i += 4
        self.b2 = p[i:i+1].reshape(1,1)

    def grad_vec(self):
        self.forward(X_XOR)
        dW1, db1, dW2, db2 = self.backward(Y_XOR)
        return np.concatenate([dW1.ravel(), db1.ravel(),
                                dW2.ravel(), db2.ravel()])

    def accuracy(self):
        preds = (self.forward(X_XOR) > 0.5).astype(int).T
        return float(np.mean(preds == Y_XOR)) * 100

# ─── Optimizers ────────────────────────────────────────────────────────────
def run_gd(lr=1.0, max_iter=2000, tol=1e-6, seed=42):
    net = MLP(seed)
    hist = [net.loss(X_XOR, Y_XOR)]
    t0 = time.perf_counter()
    for _ in range(max_iter):
        net.forward(X_XOR)
        dW1,db1,dW2,db2 = net.backward(Y_XOR)
        net.W1 -= lr*dW1; net.b1 -= lr*db1
        net.W2 -= lr*dW2; net.b2 -= lr*db2
        l = net.loss(X_XOR, Y_XOR)
        hist.append(l)
        if l < tol: break
    return hist, time.perf_counter()-t0, net

def _brent_linesearch(net, g, lo=0.0, hi=5.0, tol=1e-8, maxf=100):
    p0 = net.get_params()
    def f(eta):
        net.set_params(p0 - eta*g)
        return net.loss(X_XOR, Y_XOR)
    gold = (3 - 5**0.5)/2
    a,b = lo,hi
    x=w=v = a + gold*(b-a)
    fx=fw=fv = f(x)
    d=e=0.0
    for _ in range(maxf):
        m = 0.5*(a+b)
        tol1 = tol*abs(x)+1e-10; tol2=2*tol1
        if abs(x-m) <= tol2-0.5*(b-a): break
        p2=q2=r2=0.0
        if abs(e)>tol1:
            r2=(x-w)*(fx-fv); q2=(x-v)*(fx-fw)
            p2=(x-v)*q2-(x-w)*r2; q2=2*(q2-r2)
            if q2>0: p2=-p2
            else: q2=-q2
            r2=e; e=d
        if abs(p2)<abs(0.5*q2*r2) and p2>q2*(a-x) and p2<q2*(b-x):
            d=p2/q2; u=x+d
            if (u-a)<tol2 or (b-u)<tol2: d=tol1 if x<m else -tol1
        else:
            e=(b if x<m else a)-x; d=gold*e
        u = x+d if abs(d)>=tol1 else x+(tol1 if d>0 else -tol1)
        fu=f(u)
        if fu<=fx:
            if u<x: b=x
            else: a=x
            v,w,x=w,x,u; fv,fw,fx=fw,fx,fu
        else:
            if u<x: a=u
            else: b=u
            if fu<=fw or w==x: v,w=w,u; fv,fw=fw,fu
            elif fu<=fv or v==x or v==w: v,fv=u,fu
    net.set_params(p0)
    return x

def run_steepest(max_iter=500, tol=1e-6, seed=42):
    net = MLP(seed)
    hist = [net.loss(X_XOR, Y_XOR)]
    t0 = time.perf_counter()
    for _ in range(max_iter):
        net.forward(X_XOR)
        g = net.grad_vec()
        eta = _brent_linesearch(net, g)
        net.set_params(net.get_params() - eta*g)
        l = net.loss(X_XOR, Y_XOR)
        hist.append(l)
        if l < tol: break
    return hist, time.perf_counter()-t0, net

def run_newton(max_iter=100, tol=1e-6, eps=1e-5, seed=42):
    net = MLP(seed)
    hist = [net.loss(X_XOR, Y_XOR)]
    lam = 1.0; t0 = time.perf_counter()
    for _ in range(max_iter):
        p0 = net.get_params(); n = len(p0)
        net.forward(X_XOR); g = net.grad_vec()
        H = np.zeros((n,n))
        for i in range(n):
            for j in range(i, n):
                ei=np.zeros(n); ej=np.zeros(n); ei[i]=eps; ej[j]=eps
                net.set_params(p0+ei+ej); l1=net.loss(X_XOR,Y_XOR)
                net.set_params(p0+ei-ej); l2=net.loss(X_XOR,Y_XOR)
                net.set_params(p0-ei+ej); l3=net.loss(X_XOR,Y_XOR)
                net.set_params(p0-ei-ej); l4=net.loss(X_XOR,Y_XOR)
                H[i,j]=H[j,i]=(l1-l2-l3+l4)/(4*eps**2)
        net.set_params(p0)
        H = 0.5*(H+H.T)
        for _ in range(20):
            try:
                Hmod = H + lam*np.eye(n)
                step = np.linalg.solve(Hmod, g)
                new_p = p0 - step
                net.set_params(new_p)
                nl = net.loss(X_XOR, Y_XOR)
                if nl < hist[-1]:
                    lam = max(lam*0.5, 1e-10)
                    break
                lam *= 10
            except np.linalg.LinAlgError:
                lam *= 10
        l = net.loss(X_XOR, Y_XOR)
        hist.append(l)
        if l < tol: break
    return hist, time.perf_counter()-t0, net

# ─── Main GUI ──────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Градієнтні методи оптимізації · XOR MLP")
        self.configure(bg=BG)
        self.resizable(True, True)
        self._results = {}
        self._anim    = None
        self._build_ui()
        self.after(100, self._run_all)

    # ── layout ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── top bar
        top = tk.Frame(self, bg=BG, pady=10)
        top.pack(fill="x", padx=20)
        tk.Label(top, text="Градієнтні методи оптимізації",
                 bg=BG, fg=FG, font=("Courier New", 18, "bold")).pack(side="left")
        tk.Label(top, text="XOR · MLP 2→4→1 · MSE",
                 bg=BG, fg=FG_DIM, font=("Courier New", 11)).pack(side="left", padx=16)

        # ── controls
        ctrl = tk.Frame(self, bg=PANEL, bd=0, padx=16, pady=10)
        ctrl.pack(fill="x", padx=20, pady=(0,4))

        tk.Label(ctrl, text="seed:", bg=PANEL, fg=FG_DIM,
                 font=("Courier New",10)).grid(row=0,column=0,padx=(0,4))
        self._seed_var = tk.StringVar(value="42")
        e = tk.Entry(ctrl, textvariable=self._seed_var, width=6,
                     bg=BG, fg=FG, insertbackground=FG,
                     relief="flat", font=("Courier New",10))
        e.grid(row=0,column=1,padx=(0,16))

        tk.Label(ctrl, text="GD lr:", bg=PANEL, fg=ACCENT1,
                 font=("Courier New",10)).grid(row=0,column=2,padx=(0,4))
        self._lr_var = tk.DoubleVar(value=1.0)
        lr_sl = tk.Scale(ctrl, from_=0.05, to=2.0, resolution=0.05,
                         orient="horizontal", variable=self._lr_var,
                         bg=PANEL, fg=ACCENT1, troughcolor=BG,
                         highlightthickness=0, length=160,
                         font=("Courier New",9))
        lr_sl.grid(row=0,column=3,padx=(0,20))

        self._run_btn = tk.Button(ctrl, text="▶  Запустити",
                                  command=self._run_all,
                                  bg=ACCENT1, fg=BG,
                                  activebackground="#79b8ff", activeforeground=BG,
                                  relief="flat", font=("Courier New",10,"bold"),
                                  padx=14, pady=4, cursor="hand2")
        self._run_btn.grid(row=0,column=4)

        self._status = tk.Label(ctrl, text="Ініціалізація…",
                                bg=PANEL, fg=FG_DIM, font=("Courier New",9))
        self._status.grid(row=0,column=5,padx=16)

        # ── figure
        self._fig = plt.Figure(figsize=(14,7), facecolor=BG, tight_layout=True)
        self._fig.patch.set_facecolor(BG)
        gs = gridspec.GridSpec(2,3, figure=self._fig,
                               hspace=0.48, wspace=0.35,
                               left=0.07, right=0.97, top=0.93, bottom=0.1)
        self._ax_main  = self._fig.add_subplot(gs[0,:2])
        self._ax_log   = self._fig.add_subplot(gs[1,:2])
        self._ax_bar   = self._fig.add_subplot(gs[0,2])
        self._ax_xor   = self._fig.add_subplot(gs[1,2])
        for ax in [self._ax_main, self._ax_log, self._ax_bar, self._ax_xor]:
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values():
                sp.set_color(BORDER)
            ax.tick_params(colors=FG_DIM, labelsize=8)
            ax.xaxis.label.set_color(FG_DIM)
            ax.yaxis.label.set_color(FG_DIM)

        canvas = FigureCanvasTkAgg(self._fig, master=self)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=(0,10))
        self._canvas = canvas

        # ── stat cards
        card_frame = tk.Frame(self, bg=BG)
        card_frame.pack(fill="x", padx=20, pady=(0,14))
        self._cards = {}
        specs = [("GD",      "Градієнтний спуск",      ACCENT1),
                 ("Steepest","Метод найшвидшого спуску",ACCENT2),
                 ("Newton",  "Метод Ньютона",           ACCENT3)]
        for key, label, color in specs:
            f = tk.Frame(card_frame, bg=PANEL, padx=14, pady=8)
            f.pack(side="left", fill="x", expand=True, padx=(0,8))
            tk.Label(f, text=label, bg=PANEL, fg=color,
                     font=("Courier New",9,"bold")).pack(anchor="w")
            var_iters = tk.StringVar(value="—")
            var_loss  = tk.StringVar(value="—")
            var_time  = tk.StringVar(value="—")
            var_acc   = tk.StringVar(value="—")
            for lbl, var in [("Ітерацій:", var_iters),
                             ("Loss:",     var_loss),
                             ("Час:",      var_time),
                             ("Точність:", var_acc)]:
                row = tk.Frame(f, bg=PANEL)
                row.pack(fill="x")
                tk.Label(row, text=lbl, bg=PANEL, fg=FG_DIM,
                         font=("Courier New",8), width=10, anchor="w").pack(side="left")
                tk.Label(row, textvariable=var, bg=PANEL, fg=FG,
                         font=("Courier New",8,"bold")).pack(side="left")
            self._cards[key] = (var_iters, var_loss, var_time, var_acc)

    # ── run ─────────────────────────────────────────────────────────────────
    def _run_all(self):
        self._run_btn.config(state="disabled")
        self._status.config(text="Обчислення…")
        self.update()
        seed = int(self._seed_var.get()) if self._seed_var.get().isdigit() else 42
        lr   = self._lr_var.get()
        t = time.perf_counter()
        gd_hist,  gd_t,  gd_net  = run_gd(lr=lr, seed=seed)
        sd_hist,  sd_t,  sd_net  = run_steepest(seed=seed)
        nw_hist,  nw_t,  nw_net  = run_newton(seed=seed)
        total = time.perf_counter()-t
        self._results = {
            "GD":      (gd_hist,  gd_t,  gd_net,  ACCENT1),
            "Steepest":(sd_hist,  sd_t,  sd_net,  ACCENT2),
            "Newton":  (nw_hist,  nw_t,  nw_net,  ACCENT3),
        }
        self._update_cards()
        self._draw_plots()
        self._status.config(text=f"Готово ({total:.2f}с)")
        self._run_btn.config(state="normal")

    def _update_cards(self):
        for key,(hist,t,net,_) in self._results.items():
            vi,vl,vt,va = self._cards[key]
            vi.set(f"{len(hist)-1}")
            vl.set(f"{hist[-1]:.3e}")
            vt.set(f"{t:.3f}с")
            va.set(f"{net.accuracy():.0f}%")

    # ── plots ────────────────────────────────────────────────────────────────
    def _draw_plots(self):
        r = self._results
        # main linear
        ax = self._ax_main
        ax.cla(); ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_color(BORDER)
        ax.set_title("Збіжність (лінійна шкала)", color=FG, fontsize=10, pad=6)
        ax.set_xlabel("Ітерація", color=FG_DIM, fontsize=8)
        ax.set_ylabel("Loss (MSE)", color=FG_DIM, fontsize=8)
        ax.tick_params(colors=FG_DIM, labelsize=7)
        names = {"GD":"Градієнтний спуск","Steepest":"Метод найшвидшого спуску","Newton":"Метод Ньютона"}
        for key,(hist,t,net,color) in r.items():
            ax.plot(hist, color=color, lw=1.8, label=names[key])
        ax.legend(fontsize=8, facecolor=BG, edgecolor=BORDER, labelcolor=FG, framealpha=0.9)
        ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.6)

        # log
        ax2 = self._ax_log
        ax2.cla(); ax2.set_facecolor(PANEL)
        for sp in ax2.spines.values(): sp.set_color(BORDER)
        ax2.set_title("Збіжність (логарифмічна шкала)", color=FG, fontsize=10, pad=6)
        ax2.set_xlabel("Ітерація", color=FG_DIM, fontsize=8)
        ax2.set_ylabel("log₁₀(Loss)", color=FG_DIM, fontsize=8)
        ax2.tick_params(colors=FG_DIM, labelsize=7)
        for key,(hist,t,net,color) in r.items():
            safe = [max(v,1e-12) for v in hist]
            ax2.semilogy(safe, color=color, lw=1.8, label=names[key])
        ax2.legend(fontsize=8, facecolor=BG, edgecolor=BORDER, labelcolor=FG, framealpha=0.9)
        ax2.grid(True, which="both", color=BORDER, linewidth=0.5, alpha=0.6)
        ax2.yaxis.label.set_color(FG_DIM)

        # bar chart
        ax3 = self._ax_bar
        ax3.cla(); ax3.set_facecolor(PANEL)
        for sp in ax3.spines.values(): sp.set_color(BORDER)
        ax3.set_title("Кількість ітерацій", color=FG, fontsize=10, pad=6)
        ax3.tick_params(colors=FG_DIM, labelsize=8)
        labels = ["GD","Steepest","Newton"]
        colors = [r["GD"][3], r["Steepest"][3], r["Newton"][3]]
        vals   = [len(r[k][0])-1 for k in labels]
        bars = ax3.bar(labels, vals, color=colors, width=0.5)
        for bar,val in zip(bars,vals):
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                     str(val), ha='center', va='bottom', color=FG, fontsize=8, fontweight='bold')
        ax3.set_ylabel("Ітерацій", color=FG_DIM, fontsize=8)
        ax3.grid(True, axis="y", color=BORDER, linewidth=0.5, alpha=0.6)

        # XOR predictions heatmap (Newton result)
        ax4 = self._ax_xor
        ax4.cla(); ax4.set_facecolor(PANEL)
        for sp in ax4.spines.values(): sp.set_color(BORDER)
        ax4.set_title("XOR: передбачення (Newton)", color=FG, fontsize=10, pad=6)
        ax4.tick_params(colors=FG_DIM, labelsize=8)
        xs = np.linspace(-0.2,1.2,60)
        ys = np.linspace(-0.2,1.2,60)
        XX,YY = np.meshgrid(xs,ys)
        grid_pts = np.c_[XX.ravel(), YY.ravel()]
        net = r["Newton"][2]
        Z = net.forward(grid_pts).reshape(XX.shape)
        ax4.contourf(XX,YY,Z, levels=50, cmap="RdBu_r", alpha=0.85)
        ax4.contour(XX,YY,Z, levels=[0.5], colors=[ACCENT_Y], linewidths=1.5)
        for xi,yi,label in zip(X_XOR[:,0],X_XOR[:,1],Y_XOR.ravel()):
            color = ACCENT3 if label==0 else ACCENT2
            ax4.scatter(xi,yi,s=120,c=color,zorder=5,edgecolors=FG,linewidths=1.2)
            ax4.text(xi+0.05,yi+0.05,f"y={int(label)}", color=FG, fontsize=8)
        ax4.set_xlim(-0.2,1.2); ax4.set_ylim(-0.2,1.2)
        ax4.set_xlabel("x₁", color=FG_DIM, fontsize=8)
        ax4.set_ylabel("x₂", color=FG_DIM, fontsize=8)

        self._canvas.draw()

# ─── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.geometry("1200x820")
    app.mainloop()