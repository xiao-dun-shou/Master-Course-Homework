import numpy as np
import matplotlib.pyplot as plt

n = 50
m = 10000
S_n = np.zeros(m)
for i in range(m):          # 试验重复次数
    # 0 - 1的均匀分布, 每次抽100个数
    # x = np.random.rand(n)
    # 正态分布 N(0, 1)
    # x = np.random.randn(n)
    # 指数分布, lambda = 0.5
    x = np.random.exponential(size=n, scale=1)
    S_n[i] = np.sum(x)      # 计算这100个数的和

# mu = n * (1/2)
# mu = n * 0
mu = n * 1 / 1
# sigma = np.sqrt(n * 1/12)
# sigma = np.sqrt(n * 1)
sigma = np.sqrt(n * 1 / 1)

# count: 各个范围内的和
# bins:  返回各个bin的区间范围
# ignored：返回每个bin里面包含的数据，是一个list
count, bins, ignored = plt.hist(S_n, 20, density=True, alpha=0.5, label="S_n distribution")
plt.plot(bins,
         1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu)**2 / (2 * sigma**2)),
         linewidth=2, color='r', label="Normal distribution")
plt.legend()
plt.show()