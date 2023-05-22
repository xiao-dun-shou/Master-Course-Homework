import math
import numpy as np
import matplotlib.pyplot as plt
import random

# 生成n个自由度为k的卡方分布随机变量， return: ndarraay
def chi_square_distribution(k, n):
    Z = np.random.normal(size=(n, k))  # Z.shape = (10000, 3)
    X = np.sum(Z ** 2, axis=1)  # X.shape = (10000,) 1维度下各个元素平方和累加
    return X

# 卡方分布: 掉包
def chiDistribution_byNumpy(DOF):
    df = DOF
    # 随机数生成模块，可以生成符合自由度为DOF的卡方分布
    samples = np.random.chisquare(df, size=10000)
    # 生成密度函数的x轴数据
    x = np.linspace(0, 30, 1000)
    # 计算自由度为df的卡方分布的密度函数
    pdf = x**(df/2-1) * np.exp(-x/2) / (2**(df/2) * np.math.gamma(df/2))
    # 绘制样本直方图和卡方分布的密度函数
    plt.hist(samples, bins=50, density=True, alpha=0.5, label='samples')
    plt.plot(x, pdf, label='chi2 pdf')
    plt.legend()
    plt.show()

# 卡方分布: 不掉包
def chiDistribution(DOF, n=10000):
    """
    DOF: degree of freedom 自由度
    n:   随机变量的数量，越大越能模拟真实分布
    """
    # 绘制卡方分布的概率密度函数
    def chi_square_pdf(x, k):
        return x ** (k / 2 - 1) * np.exp(-x / 2) / (2 ** (k / 2) * np.math.gamma(k / 2))

    # 生成自由度为3的卡方分布随机变量
    X = chi_square_distribution(DOF, n)

    # 绘制卡方分布的概率密度函数
    x = np.linspace(0, 20, 1000)
    y = chi_square_pdf(x, DOF)
    plt.plot(x, y, 'r-', lw=2, label='[fact]Chi-square distribution probability density function')

    # 绘制直方图
    plt.hist(X, bins=int(math.sqrt(n)), density=True, alpha=0.5, label='[theory]Sample histogram of Chi-squared distribution')
    plt.legend()
    plt.show()

# t分布: 不掉包
def tDistribution(DOF, n=10000):
    # 模拟t分布
    def t_distribution(DOF, n=10000):
        u = np.random.normal(size=(n, ))
        v = chi_square_distribution(DOF, n)
        return u / np.sqrt(v / DOF)

    # t分布概率密度函数
    def t_pdf(x, n):
        numerator = math.gamma((n + 1) / 2)
        denominator = math.sqrt(n * math.pi) * math.gamma(n / 2)
        return numerator / denominator * (1 + x ** 2 / n) ** (-(n + 1) / 2)

    # 生成样本数据
    sample_data = t_distribution(DOF, n)

    # 绘制直方图
    plt.hist(sample_data, bins=int(math.sqrt(n)), density=True, alpha=0.5, color='b')
    plt.title('t Distribution DOF = ' + str(DOF))
    plt.xlabel('x')
    plt.ylabel('Probability density')

    # 绘制理论上的t分布曲线
    x = [i / 10 for i in range(-100, 101)]
    y = [t_pdf(i, DOF) for i in x]
    plt.plot(x, y, color='r')
    plt.show()

# F分布: 不掉包
def fDistribution(mDOF=10, nDOF=10, n=10000, boundnum=20):
    # 模拟f分布
    def f_distribution(mDOF=1, nDOF=1, n=10000):
        M = chi_square_distribution(mDOF, n)
        N = chi_square_distribution(nDOF, n)
        K = (M/mDOF) / (N/nDOF)
        K[K>boundnum] = 0
        return K

    # F分布概率密度函数
    def F_pdf(x, df1, df2):
        numerator = math.gamma((df1 + df2) / 2) * (df1 / df2) ** (df1 / 2) * x ** ((df1 / 2) - 1)
        denominator = math.gamma(df1 / 2) * math.gamma(df2 / 2) * (1 + (df1 / df2) * x) ** ((df1 + df2) / 2)
        return numerator / denominator

    # 生成样本数据
    sample_data = f_distribution(mDOF, nDOF, n)

    # 绘制直方图
    plt.hist(sample_data, bins=int(math.sqrt(n)), density=True, alpha=0.5, color='b')
    plt.title('t Distribution m,n = ' + str(mDOF) + " " + str(nDOF) )
    plt.xlabel('x')
    plt.ylabel('Probability density')

    # 绘制理论上的t分布曲线
    x = [i / 10 for i in range(0, boundnum * 10)]
    y = [F_pdf(i, mDOF, nDOF) for i in x]
    plt.plot(x, y, color='r')
    plt.show()
    plt.show()

if __name__ == "__main__":
    # chiDistribution(3, 100000)
    # tDistribution(10)
    fDistribution(3, 2, 10000)