import numpy as np
import matplotlib.pyplot as plt

# 定义随机变量序列X
X = np.random.randint(1, 7, size=(10000, 10))
# 计算样本均值序列的标准差
s = np.std(np.mean(X, axis=1))

# 按照切比雪夫大数定律计算界
epsilons = np.linspace(0.01, 0.5, 100)
bounds = s / np.sqrt(epsilons)

# 绘制界的图像
plt.plot(epsilons, bounds, label='Chebyshev Bound')
plt.xlabel('Epsilon')
plt.ylabel('Bound')
plt.legend()
plt.show()