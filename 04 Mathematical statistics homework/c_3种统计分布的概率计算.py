"""
本程序将根据分位数点计算正态分布、卡方分布、t分布、F分布

使用示例：
print(normCalculateP(0.08))
print(chiCalculateP(0.0201, 2))
print(fCalculateP(2.4163, 10, 9))
print(tCalculateP(9.9248, 2))

print results:
0.5318813720139874
0.0099996675049777
0.8999984008482127
0.9949999571224089
可以查阅教材附表进行对照，结果是合理的，误差很小。
"""

from scipy.stats import f,chi2,t,norm

# 正态分布
def normCalculateP(x, mu=0, sigma=1):
    return norm.cdf(x, mu, sigma)

def normCalculateX(p, mu=0, sigma=1):
    # 计算概率密度函数
    return norm.ppf(p, mu, sigma)



# 卡方分布
def chiCalculateP(x, n):
    # 计算P(χ²(n) ≤ x)
    p = chi2.cdf(x, df=n)
    return p

def chiCalculateX(p, n):
    # 计算P(χ²(n) ≤ x)
    # 已知p, n, 求x
    x = chi2.ppf(p, df=n)
    return x

# t分布
def tCalculateP(x, n):
    # 计算P(F(m, n) ≤ x)
    return t.cdf(x, n)

def tCalculateX(p, n):
    # 计算累积分布函数的逆函数
    return t.ppf(p, n)

# F分布
def fCalculateP(x, m, n):
    # 计算P(F(m, n) ≤ x)
    # 已知x, m, n 求 p
    return f.cdf(x, m, n)

def fCalculateX(p, m, n):
    # 计算P(F(m, n) ≤ x)
    # 已知p, m, n 求 x
    return f.ppf(p, m, n)

if __name__ == "__main__":
    # print(fCalculateX(0.05, 8, 11))
    # print((1 - chiCalculateP(28.70, 25))*2)
    # print(chiCalculateP(108.87, 69), chiCalculateP(55.59, 69), chiCalculateP(108.87, 69) - chiCalculateP(55.59, 69))
    # print(tCalculateP(0.3477, 19))
    # print(chiCalculateX(0.025, 69)*1.16)
    # print(normCalculateP(4.4898) - normCalculateP(0.5698))
    # print(tCalculateX(0.01, 162))
    # print(1 - fCalculateP(1.79, 2, 2))
    print(1-fCalculateP(9.733, 2, 4))