import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 生成随机数
np.random.seed(123)
x = np.random.normal(0, 1, 1000)

# 绘制经验分布函数
fig, ax = plt.subplots()
# cumulative=True --> 累计频率
# density=True    --> 将直方图的纵轴单位改为概率密度函数
_, bins, _ = ax.hist(x, bins=40, density=True, cumulative=True, label='Empirical distribution')

# 计算正态分布的CDF, 分布函数
mu, sigma = norm.fit(x)     # 样本的均值和标准差
norm_cdf = norm.cdf(bins, mu, sigma)    # 标准库函数，可以计算正态分布的累积分布函数
ax.plot(bins, norm_cdf, 'r--', label='Normal distribution')

ax.legend(loc='right')
ax.set_title('Empirical distribution vs. Normal distribution')
plt.show()

