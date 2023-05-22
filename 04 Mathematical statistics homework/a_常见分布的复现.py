import math

import numpy as np
import matplotlib.pyplot as plt
import scipy        # 高级科学计算库, 他会操作Numpy

# 离散型：[单点分布]
def PointDeltaDistribution():
    # 设置参数
    x = [0]
    y = [1]
    title = 'Point Delta Distribution'
    type = 1
    return x, y, title, type

# 离散型：[两点分布]， 伯努利实验
def TwoPointDeltaDistribution(probability=0.5):
    # 设置参数
    x = [0, 1]
    y = [probability, 1 - probability]
    title = 'Two-point Distribution'
    type = 1
    return x, y, title, type

# 离散型：[均匀分布]，古典概型
def UniformDistrbution(N=10):
    n = N
    x = np.arange(0, N)
    y = np.ones_like(x) / n
    title = 'Uniform Distribution'
    type = 1
    return x, y, title, type

# 离散型：[几何分布], 事件在第X次试验中首次发生
def GeoDistrbution(probability=0.5, X=11):
    p = probability
    x = np.arange(1, X)
    y = p * (1 - p) ** (x - 1)
    title = 'Geometric Distribution'
    type = 1
    return x, y, title, type

# 离散型：[二项分布], 事件在N重伯努利实验中成功的次数X
def BinomialDistrbution(probability=0.5, N=50, X=50):
    n = N
    p = probability
    x = np.arange(0., X)
    y = np.zeros_like(x)
    for i in range(X):
        # 排列组合的算法：np.math.comb(n, i) --> C_n^i
        y[i] = np.math.comb(n, i) * p ** i * (1 - p) ** (n - i)
    title = 'Binomial Distribution'
    type = 1
    return x, y, title, type

# 离散型：[泊松分布], 在某段时间内事件A发生的次数X
def PoissonDistrbution(lam=10, N=50):
    # 定义泊松分布概率质量函数
    def poisson_pmf(lam, k):
        return (lam ** k) * np.exp(-lam) / np.math.factorial(k)
    # 设置参数
    n = N
    x = np.arange(0., n)
    y = np.zeros_like(x)
    y = [poisson_pmf(lam, k) for k in x]
    title = 'Poisson Distribution'
    type = 1
    return x, y, title, type

# 离散型：[超几何分布], 在含M个次品的N个产品中，一次性抽取n个，抽出的次品个数X
def HypergeometricDistribution(N=100, M=30, n=30):
    # 设置参数
    x = np.arange(max(0, n - (N - M)), min(n, M) + 1)
    y = np.zeros_like(x)
    y = y.astype(np.float32)
    for i in range(len(x)):
        y[i] = np.math.comb(M, x[i]) * np.math.comb(N - M, n - x[i]) / np.math.comb(N, n)
    title = 'Hypergeometric Distribution'
    type = 1
    return x, y, title, type

# 连续型: [指数分布]
def ExponentialDistribution(lam = 0.5):
    def exponential_pdf(x, lambd):
        y = lambd * np.exp(-lambd * x)
        y[x < 0] = 0
        return y
    # 设置参数
    x = np.linspace(0, 10, 100)
    y = exponential_pdf(x, lam)
    title = 'Exponential Distribution'
    type = 2
    return x, y, title, type

# 连续型: [均匀分布]
def UniformDistribution_(a=0,b=5):
    x = np.linspace(a, b, 100)
    y = np.ones_like(x) / (b - a)
    title = 'Uniform Distribution'
    type = 2
    return x, y, title, type

# 连续型: [正态分布]
def NormalDistrbution(mu = 0, sigma = 1):
    def normal_pdf(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    # 设置参数
    x = np.linspace(-4, 4, 100)
    y = normal_pdf(x, mu, sigma)
    title = 'Normal Distribution'
    type = 2
    return x, y, title, type

# 主函数
def main(x, y, title, type):
    if type == 1:
        plt.stem(x, y)
    elif type == 2:
        plt.plot(x, y)
    else:
        raise Exception("出现了一个非法的type")
    plt.title(title)
    plt.xlabel('x, Random variables events')
    plt.ylabel('p(x), Probability')
    plt.show()

cfg = {
    "单点分布" : PointDeltaDistribution,
    "两点分布" : TwoPointDeltaDistribution,
    "古典概型" : UniformDistrbution,
    "几何分布" : GeoDistrbution,
    "二项分布" : BinomialDistrbution,
    "泊松分布" : PoissonDistrbution,
    "超几何分布" : HypergeometricDistribution,
    "指数分布" : ExponentialDistribution,
    "正态分布" : NormalDistrbution,
    "均匀分布" : UniformDistribution_,
}

if __name__ == "__main__":
    model = "几何分布"
    x, y, title, type = cfg[model]()
    main(x, y, title, type)


