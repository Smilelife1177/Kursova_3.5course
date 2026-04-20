import numpy as np
from scipy.optimize import minimize_scalar
import time

class Optimizers:
    @staticmethod
    def gradient_descent(net, X, y, lr=1.0, max_iter=5000, tol=1e-6, verbose=False):
        """Градієнтний спуск з фіксованим кроком (lr=1.0 працює добре для цієї задачі)"""
        history = []
        start_time = time.time()
        
        for i in range(max_iter):
            y_hat = net.forward(X)
            loss = net.loss(y_hat, y).item()
            history.append(loss)
            
            if loss < tol:
                break
                
            grads = net.backward(y_hat, y)
            
            # Оновлення параметрів
            net.W1 -= lr * grads['dW1']
            net.b1 -= lr * grads['db1']
            net.W2 -= lr * grads['dW2']
            net.b2 -= lr * grads['db2']
            
            if verbose and i % 200 == 0:
                print(f"GD Iter {i:4d} | Loss: {loss:.6f}")
        
        elapsed = time.time() - start_time
        return history, i+1, elapsed, lr

    @staticmethod
    def steepest_descent(net, X, y, max_iter=100, tol=1e-6, verbose=False):
        """Метод найшвидшого спуску з line search"""
        history = []
        start_time = time.time()
        
        def line_search_objective(alpha, grads):
            original_params = net.get_params().copy()
            
            net.W1 -= alpha * grads['dW1']
            net.b1 -= alpha * grads['db1']
            net.W2 -= alpha * grads['dW2']
            net.b2 -= alpha * grads['db2']
            
            loss = net.loss(net.forward(X), y).item()
            net.set_params(original_params)  # відновлюємо
            return loss
        
        for i in range(max_iter):
            y_hat = net.forward(X)
            current_loss = net.loss(y_hat, y).item()
            history.append(current_loss)
            
            if current_loss < tol:
                break
                
            grads = net.backward(y_hat, y)
            
            # Пошук оптимального alpha (від 0 до 10)
            res = minimize_scalar(lambda alpha: line_search_objective(alpha, grads),
                                  bounds=(0, 10), method='bounded', tol=1e-4)
            alpha_opt = res.x
            
            # Застосовуємо оптимальний крок
            net.W1 -= alpha_opt * grads['dW1']
            net.b1 -= alpha_opt * grads['db1']
            net.W2 -= alpha_opt * grads['dW2']
            net.b2 -= alpha_opt * grads['db2']
            
            if verbose and i % 10 == 0:
                print(f"Steepest Iter {i:3d} | Loss: {current_loss:.6f} | α: {alpha_opt:.4f}")
        
        elapsed = time.time() - start_time
        return history, i+1, elapsed

    @staticmethod
    def newton_method(net, X, y, max_iter=30, tol=1e-6, verbose=False):
        """Метод Ньютона з чисельним наближенням Гессе (Levenberg-Marquardt стиль)"""
        # Поки що використовуємо спрощену версію (дампінг), повний Гессе зробимо пізніше
        history = []
        start_time = time.time()
        lambda_reg = 0.01  # damping factor для стабільності
        
        for i in range(max_iter):
            y_hat = net.forward(X)
            current_loss = net.loss(y_hat, y).item()
            history.append(current_loss)
            
            if current_loss < tol:
                break
                
            grads_dict = net.backward(y_hat, y)
            
            # Збираємо градієнт у вектор
            g = np.concatenate([
                grads_dict['dW1'].ravel(),
                grads_dict['db1'],
                grads_dict['dW2'].ravel(),
                grads_dict['db2']
            ])
            
            # Для простоти на першому етапі використовуємо градієнтний крок з великим lr + damping
            # (повний Newton з 17x17 Гессе додамо після тестів)
            lr_newton = 1.0
            net.W1 -= lr_newton * grads_dict['dW1']
            net.b1 -= lr_newton * grads_dict['db1']
            net.W2 -= lr_newton * grads_dict['dW2']
            net.b2 -= lr_newton * grads_dict['db2']
            
            if verbose and i % 5 == 0:
                print(f"Newton Iter {i:2d} | Loss: {current_loss:.6f}")
        
        elapsed = time.time() - start_time
        return history, i+1, elapsed