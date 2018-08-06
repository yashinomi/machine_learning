import numpy as np
import matplotlib.pyplot as plt
from modules.dataset import dataset2

hyper_lambda = 10

np.random.seed(42)

# データセットの初期化
X_train, Y_train = dataset2()
Y_diag = np.diag(Y_train)
dim_Y = len(Y_train)
Y_train = Y_train.reshape(1, -1)
K = Y_diag @ X_train @ X_train.T @ Y_diag


# -1倍した双対ラグランジュ
def minus_dual_lagrange(alpha):
    result = (alpha @ K @ alpha)/(-4*hyper_lambda) \
             + alpha @ np.ones(len(alpha))
    return -result


# -1倍した双対ラグランジュをalphaで微分した関数
def div_minus_dual_lagrange(alpha: np.array) -> np.array:
    result = (K @ alpha) / (2 * hyper_lambda)
    result -= np.ones(len(result))
    return result


# 元の最適化する関数J(w)
def loss_and_regularize(w):
    result = np.maximum(np.zeros(dim_Y), np.ones(dim_Y) - np.dot(X_train, w))
    result = np.dot(result, np.ones(len(result)))
    result += hyper_lambda * (np.dot(w, w))
    return result


# アルミホの条件を元に学習率を求める
def learning_rate(alpha: np.array, div: np.array) -> float:
    eta = 0.5
    rho = 0.5
    c1 = 0.001

    while minus_dual_lagrange(alpha - eta * div) > minus_dual_lagrange(alpha) - c1 * np.dot(div, div):
        eta = rho * eta
        print(eta)
    return eta


# 次のalphaを求める
def projected(alpha):
    div = div_minus_dual_lagrange(alpha)
    result = alpha - learning_rate(alpha, div) * div
    result[result > 1] = 1
    result[result < 0] = 0
    return result


def alpha_to_w(alpha):
    return alpha @ Y_diag @ X_train / (hyper_lambda * 2)


# パラメータの初期化
alpha_0 = np.random.rand(dim_Y)
alpha_1 = projected(alpha_0)
dl = -minus_dual_lagrange(alpha_0)
loss = loss_and_regularize(alpha_to_w(alpha_0))
plot_lagrange = [dl]
plot_loss = [loss]

# alphaが収束するまで繰り返す
while (alpha_1 != alpha_0).any():
    alpha_0 = alpha_1
    alpha_1 = projected(alpha_0)
    dl = -minus_dual_lagrange(alpha_0)
    loss = loss_and_regularize(alpha_to_w(alpha_0))
    plot_lagrange.append(dl)
    plot_loss.append(loss)

# 双対ラグランジュ
plt.plot(plot_lagrange)
plt.title("Dual Lagrange")
plt.show()

plt.plot(plot_loss)
plt.title("Sum of Hinge Loss")
plt.show()

opted_w = alpha_to_w(alpha_0)
plt.scatter(X_train @ np.array([1, 0]), X_train @ np.array([0, 1]), c=Y_train.T.ravel())
X_scatter = np.array(range(-2, 3))
Y_scatter = [opted_w[0] / opted_w[1] * x for x in X_scatter]
plt.plot(X_scatter, Y_scatter)
plt.title("Training Data set")
plt.show()
