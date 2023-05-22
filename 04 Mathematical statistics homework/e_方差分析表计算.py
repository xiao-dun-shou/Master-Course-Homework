"""
K = [[39.50, 69.75, 55.30], [49.60, 55.058, 59.90], [61.10, 49.10, 54.35], [54.70, 59.45, 50.40]]

"""
from scipy.stats import f

class varTable:
    def __init__(self, y, K, m, r, nullFlag=-1):
        self.y = y              # 实验结果y
        self.K = K              # K 按列存放
        self.m = len(K)         # 实验的重复次数
        self.r = r              # 水平个数
        self.totalDegree = len(y) - 1  # 总自由度, or = self.m * self.r
        self.f_degree = r - 1   # 每一列的自由度
        self.nullFlag = nullFlag # 空列位置标记。默认为最后一个元素
        #########################
        self.S2_j = self.calculate_S2_j()          # 第j列的离差平方和
        self.MSE = self.calculate_MSE()            # 均方差
        self.f = self.calculate_f()             # f统计量
        self.p = self.calculate_p()             # 统计量对应的p - value

    def calculate_S2_j(self):
        Kmean = []              # 保存K的均值
        result = []
        for k in self.K:
            Kmean.append(sum(k) / len(k))
        for idx,k in enumerate(self.K):
            temp = 0
            for i in k:
                temp += (Kmean[idx] - i) ** 2
            result.append(temp / self.r)
        return result

    def caicluate_total_MSE(self):
        y_mean = sum(self.y) / len(self.y)
        result = 0
        for i in self.y:
            result += (i - y_mean) ** 2
        print("总离差平方和S_T^2 = ", result)
        print("离差平方和是否符合加法原则: ", (result - sum(self.S2_j)) ** 2 < 1)

    def calculate_MSE(self):
        result = []
        degree = []
        for i in range(len(self.K)):
            if self.K[i] == self.K[self.nullFlag]:
                # 误差位
                degree.append(self.totalDegree - (self.r - 1) * (len(self.S2_j) - 1))
            else:
                degree.append(self.r - 1)
        for idx, i in enumerate(degree):
            result.append(self.S2_j[idx] / i)
        return result

    def calculate_f(self):
        result = []
        for i in self.MSE:
            result.append(i / self.MSE[self.nullFlag])
        return result

    def calculate_p(self):
        result = []
        for x in self.f:
            result.append(1 - f.cdf(x, self.r - 1, self.totalDegree - (self.r - 1) * (len(self.S2_j) - 1)))
        return result

    def printValues(self):
        print("S_j^2 = ", self.S2_j)
        print("MSE   = ", self.MSE)
        print("f     = ", self.f)
        print("p     = ", self.p)

if __name__ == "__main__":
    # 对应教材P262， 表6.4.12 => 可复现
    # y = [86, 95, 91, 94, 91, 96, 83, 88]
    # K = [[366, 358], [368, 356], [352, 372], [351, 373], [361, 363], [359, 365], [359, 365]]

    # 对应视频 => 不可复现 => 怀疑视频出错了
    y = [82, 78, 76, 85, 83, 86, 92, 79]
    K = [[321, 340], [329, 332], [331, 330], [333, 328], [323,338], [329, 332], [345, 316]]

    # 对应视频 => 可复现
    # y = [ 13.45, 12.85, 13.20, 18.10, 23.10, 28.55, 18.05, 19.10, 18.15]
    # K = [[39.50, 69.75, 55.30], [49.60, 55.058, 59.90], [61.10, 49.10, 54.35], [54.70, 59.45, 50.40]]
    r = len(K[0])   # 水平数: 1 or 2
    m = 4           # 重复次数: 一列中有多少个1 or 2
    vt = varTable(y, K, m, r, nullFlag=-1)
    vt.printValues()
    vt.caicluate_total_MSE()




