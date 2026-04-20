import numpy as np
from scipy.optimize import minimize_scalar
import time


class Optimizers:

    # ------------------------------------------------------------------ #
    #  1. Градієнтний спуск з фіксованим кроком                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def gradient_descent(net, X, y, lr=1.0, max_iter=5000, tol=1e-6, verbose=False):
        """
        Градієнтний спуск:
            θ_{k+1} = θ_k − lr · ∇L(θ_k)

        Крок lr фіксований на всіх ітераціях.
        """
        history = []
        start_time = time.time()

        for i in range(max_iter):
            y_hat = net.forward(X)
            loss = net.loss(y_hat, y)
            history.append(float(loss))

            if loss < tol:
                break

            grads = net.backward(y_hat, y)
            net.W1 -= lr * grads['dW1']
            net.b1 -= lr * grads['db1']
            net.W2 -= lr * grads['dW2']
            net.b2 -= lr * grads['db2']

            if verbose and i % 200 == 0:
                print(f"  GD Iter {i:4d} | Loss: {loss:.6f}")

        elapsed = time.time() - start_time
        return history, i + 1, elapsed, lr

    # ------------------------------------------------------------------ #
    #  2. Метод найшвидшого спуску (точний line search)                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def steepest_descent(net, X, y, max_iter=200, tol=1e-6, verbose=False):
        """
        Метод найшвидшого спуску:
            α_k = argmin_{α≥0} L(θ_k − α · ∇L(θ_k))
            θ_{k+1} = θ_k − α_k · ∇L(θ_k)

        Оптимальний крок α шукається точним golden-section search.
        """
        history = []
        start_time = time.time()

        def line_search_objective(alpha, params_before, grads):
            """Обчислює L при кроці alpha від поточної точки."""
            candidate = params_before - alpha * grads
            net.set_params(candidate)
            loss = float(net.loss(net.forward(X), y))
            return loss

        for i in range(max_iter):
            y_hat = net.forward(X)
            current_loss = float(net.loss(y_hat, y))
            history.append(current_loss)

            if current_loss < tol:
                break

            grads_dict = net.backward(y_hat, y)

            # Зберігаємо поточні параметри і градієнт як вектори
            params_now = net.get_params().copy()
            g = np.concatenate([
                grads_dict['dW1'].ravel(),
                grads_dict['db1'],
                grads_dict['dW2'].ravel(),
                grads_dict['db2']
            ])

            # Точний пошук α* ∈ [0, 10]
            res = minimize_scalar(
                lambda alpha: line_search_objective(alpha, params_now, g),
                bounds=(0, 10), method='bounded', options={'xatol': 1e-8}
            )
            alpha_opt = res.x

            # Застосовуємо оптимальний крок
            net.set_params(params_now - alpha_opt * g)

            if verbose and i % 10 == 0:
                print(f"  Steepest Iter {i:3d} | Loss: {current_loss:.6f} | α*: {alpha_opt:.5f}")

        elapsed = time.time() - start_time
        return history, i + 1, elapsed

    # ------------------------------------------------------------------ #
    #  3. Метод Ньютона з чисельною матрицею Гесса                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def newton_method(net, X, y, max_iter=50, tol=1e-6,
                      eps=1e-5, lambda_init=1e-3, verbose=False):
        """
        Метод Ньютона:
            H(θ_k) · d_k = ∇L(θ_k)
            θ_{k+1} = θ_k − d_k

        Матриця Гесса H обчислюється чисельно методом скінченних різниць
        (central differences), розмір 17×17.

        Levenberg–Marquardt демпфування:
            (H + λI) · d = g
        забезпечує збіжність навіть коли H не є позитивно визначеною.
        λ адаптивно зменшується при успішних кроках і збільшується при невдалих.
        """
        history = []
        start_time = time.time()
        lambda_lm = lambda_init     # початковий коефіцієнт демпфування

        def compute_loss(params):
            """Допоміжна: обчислює скаляр loss для вектора параметрів."""
            net.set_params(params)
            return float(net.loss(net.forward(X), y))

        def compute_gradient(params):
            """
            Градієнт методом центральних різниць:
                ∂L/∂θ_i ≈ [L(θ + ε·eᵢ) − L(θ − ε·eᵢ)] / (2ε)
            Точніший за forward differences: похибка O(ε²) замість O(ε).
            """
            n = len(params)
            grad = np.zeros(n)
            for j in range(n):
                p_plus  = params.copy(); p_plus[j]  += eps
                p_minus = params.copy(); p_minus[j] -= eps
                grad[j] = (compute_loss(p_plus) - compute_loss(p_minus)) / (2 * eps)
            return grad

        def compute_hessian(params, grad):
            """
            Матриця Гесса методом центральних різниць по градієнту:
                ∂²L/∂θ_i∂θ_j ≈ [∇L(θ + ε·eⱼ) − ∇L(θ − ε·eⱼ)]_i / (2ε)

            Розмір: (17, 17).  Симетризуємо H = (H + Hᵀ)/2 для числової стійкості.
            """
            n = len(params)
            H = np.zeros((n, n))
            for j in range(n):
                p_plus  = params.copy(); p_plus[j]  += eps
                p_minus = params.copy(); p_minus[j] -= eps
                g_plus  = compute_gradient(p_plus)
                g_minus = compute_gradient(p_minus)
                H[:, j] = (g_plus - g_minus) / (2 * eps)
            H = (H + H.T) / 2          # гарантуємо симетрію
            return H

        for i in range(max_iter):
            params = net.get_params().copy()
            current_loss = compute_loss(params)
            history.append(current_loss)

            if current_loss < tol:
                break

            # 1. Градієнт (аналітичний через backprop — швидше і точніше)
            y_hat = net.forward(X)
            grads_dict = net.backward(y_hat, y)
            g = np.concatenate([
                grads_dict['dW1'].ravel(),
                grads_dict['db1'],
                grads_dict['dW2'].ravel(),
                grads_dict['db2']
            ])

            # 2. Матриця Гесса (чисельно)
            H = compute_hessian(params, g)

            # 3. Levenberg–Marquardt: розв'язуємо (H + λI)·d = g
            #    Адаптивно підбираємо λ, щоб крок дійсно зменшував loss
            step_accepted = False
            for _ in range(10):                          # макс. 10 спроб
                H_reg = H + lambda_lm * np.eye(len(g))
                try:
                    d = np.linalg.solve(H_reg, g)        # Newton direction
                except np.linalg.LinAlgError:
                    lambda_lm *= 10
                    continue

                new_params = params - d
                new_loss   = compute_loss(new_params)

                if new_loss < current_loss:              # крок успішний
                    net.set_params(new_params)
                    lambda_lm = max(lambda_lm / 3, 1e-10)   # зменшуємо λ
                    step_accepted = True
                    break
                else:
                    lambda_lm = min(lambda_lm * 10, 1e10)   # збільшуємо λ

            if not step_accepted:
                # fallback: маленький градієнтний крок
                net.set_params(params - 1e-3 * g)

            if verbose:
                eigs = np.linalg.eigvalsh(H)
                status = "✓" if step_accepted else "↩ fallback"
                print(f"  Newton Iter {i:2d} | Loss: {current_loss:.6f} | "
                      f"λ: {lambda_lm:.2e} | "
                      f"min(eig(H)): {eigs.min():.3e} | {status}")

        elapsed = time.time() - start_time
        return history, i + 1, elapsed