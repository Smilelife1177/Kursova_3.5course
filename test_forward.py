from neural_network import MLP
from xor_data import X, y

net = MLP(seed=42)

# Тест forward pass
y_hat = net.forward(X)
loss_val = net.loss(y_hat, y)
print(f"Initial loss: {loss_val:.4f}")
print(f"Predictions:\n{y_hat.round(3)}")

# Тест backward pass
grads = net.backward(y_hat, y)
print(f"\ndW1 shape: {grads['dW1'].shape}")  # (2, 4)
print(f"dW2 shape: {grads['dW2'].shape}")  # (4, 1)

# Тест get/set params (для методу Ньютона)
params = net.get_params()
print(f"\nTotal params: {len(params)}")  # 8+4+4+1 = 17

# Очікуваний вивід:
# Initial loss: ~0.25
# Total params: 17