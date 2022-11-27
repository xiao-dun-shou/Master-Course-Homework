from Hungary_Algorithm_ import Hungarian
import numpy as np
import matplotlib.pyplot as plt

# 测试代码。z主要是为了监控my_cav的变换
# my_cav = []

# 初始参数设置
# 工作负荷矩阵 验证论文第四章第一个图 7*7的矩阵
WorkloadMatrtix2 = [
    [9,11,3,6,6,7,4],
    [10,9,11,5,6,3,6],
    [8,10,5,6,4,7,1],
    [6,8,10,4,9,5,8],
    [11,1,9,8,7,11,8],
    [4,1,3,2,7,0,3],
    [4,6,2,9,8,5,4],
]
# 验证表2第三行
WorkloadMatrtix = [
    [49, 56, 53, 82, 35, 53, 76, 86, 46, 32, 33, 41, 53, 43, 49, 73, 83, 43, 83, 36],
    [78, 56, 46, 38, 39, 73, 69, 80, 66, 54, 82, 39, 34, 49, 71, 38, 65, 39, 37, 44],
    [60, 63, 62, 51, 71, 86, 73, 61, 37, 37, 35, 67, 85, 75, 76, 71, 86, 64, 32, 72],
    [40, 67, 72, 74, 39, 49, 49, 69, 85, 73, 81, 56, 64, 38, 32, 52, 88, 76, 52, 36],
    [57, 69, 63, 82, 68, 31, 55, 40, 82, 40, 38, 60, 86, 83, 88, 63, 84, 59, 47, 80],
    [83, 38, 66, 77, 85, 31, 54, 67, 83, 35, 31, 61, 46, 58, 68, 41, 76, 46, 47, 76],
    [72, 32, 39, 46, 88, 51, 34, 88, 70, 79, 79, 84, 81, 61, 38, 83, 30, 81, 52, 37],
    [59, 64, 40, 81, 82, 84, 59, 52, 84, 39, 64, 74, 36, 40, 39, 34, 87, 47, 66, 83],
    [77, 46, 58, 68, 45, 78, 50, 70, 44, 50, 71, 80, 74, 57, 39, 72, 69, 31, 38, 55],
    [86, 81, 75, 60, 78, 65, 50, 55, 81, 64, 74, 50, 71, 78, 51, 49, 43, 72, 67, 68],
    [45, 83, 51, 42, 50, 37, 63, 32, 44, 83, 87, 72, 88, 40, 31, 88, 84, 51, 60, 83],
    [77, 46, 67, 56, 51, 88, 75, 85, 60, 83, 57, 46, 49, 68, 53, 85, 65, 80, 78, 40],
    [33, 40, 48, 40, 44, 78, 33, 42, 62, 63, 76, 89, 36, 67, 41, 65, 83, 62, 82, 65],
    [72, 63, 39, 84, 33, 76, 61, 82, 73, 79, 34, 38, 38, 86, 56, 33, 62, 33, 83, 45],
    [65, 76, 64, 51, 77, 36, 39, 80, 58, 53, 79, 48, 71, 30, 33, 65, 57, 83, 55, 60],
    [52, 60, 56, 44, 37, 74, 73, 67, 71, 63, 38, 44, 83, 67, 79, 35, 32, 88, 48, 67],
    [59, 78, 31, 83, 42, 75, 71, 53, 88, 84, 86, 46, 49, 31, 46, 73, 59, 84, 89, 46],
    [55, 56, 59, 77, 57, 75, 61, 56, 43, 75, 88, 66, 88, 79, 70, 40, 85, 70, 58, 80],
    [38, 86, 79, 55, 58, 34, 77, 30, 39, 36, 44, 82, 51, 70, 48, 63, 41, 82, 51, 71],
    [81, 48, 75, 87, 38, 86, 63, 56, 62, 57, 60, 73, 38, 67, 58, 31, 38, 86, 31, 80]]
# 代理数目
N = np.array(WorkloadMatrtix, dtype=np.int32).shape[0]
# 步长补偿
step_compensation = 0.01 * N            # 作者没给出超参数分析，我自行设置的
# 收敛判断常数
gamma = 0.01 * N                        # 作者没给出超参数分析，我自行设置的

def demo():
    """
    测试用例
    :return:
    """
    profit_matrix = [
        [62, 75, 80, 93, 0, 97],
        [75, 0, 82, 85, 71, 97],
        [80, 75, 81, 0, 90, 97],
        [78, 82, 0, 80, 50, 98],
        [0, 85, 85, 80, 85, 99],
        [65, 75, 80, 75, 68, 0]]
    hungarian = Hungarian(profit_matrix, is_profit_matrix=True)
    hungarian.calculate()
    print("Expected value:\t\t523")
    print("Calculated value:\t", hungarian.get_total_potential())  # = 523
    print("Expected results:\n\t[(0, 3), (2, 4), (3, 0), (5, 2), (1, 5), (4, 1)]")
    print("Results:\n\t", hungarian.get_results())
    print("-" * 80)

class FAP():
    """
    均值逼近算法的核心代码
    """
    def __init__(self, Matrtix, N=None, step=1, gamma=0.1):
        """
        :param Matrtix: ndarray / list 均可
        :param N:       int
        :param step:    num
        :param gamma:   num
        """

        self.WorkloadMatrtix = Matrtix
        if isinstance(self.WorkloadMatrtix, np.ndarray):
            self.wlMatrtix = self.WorkloadMatrtix
        else:
            self.wlMatrtix = np.array(self.WorkloadMatrtix, dtype=np.int32)
        if N == None:
            self.N = np.array(WorkloadMatrtix, dtype=np.int32).shape[0]
        else:
            self.N = N
        self.step_compensation = step       # 作者没给出超参数分析，我自行设置的
        self.gamma = gamma                  # 作者没给出超参数分析，我自行设置的
        self.delta = 0                      # 步长
        self.c_av_min = 0                   # c_av 最大最小值
        self.c_av_max = 0                   # c_av 最大最小值

        # 保存计算结果
        self.result_plt_y = []               # 保存每次收敛的均值
        self.result_plt_x = []
        self.result_min = 999999             # 保存当前的最小均值
        self.result_x = []                   # 保存最小均值对应的可行解
        self.result_ave = 0                  # 保存最终的工作负荷均值

    def calculateCav(self, li):
        """
        本函数用于计算某个可行解下的平均工作负荷
        :param li: 某个可行解
        :return:
        """
        # print('正在计算新的平均工作负荷：')
        # print('当前解',li)
        # print(wlMatrtix)
        # 求平均值
        value = 0
        for row, column in li:
            # print(row, column, wlMatrtix[row, column])
            value += self.wlMatrtix[row, column]
        ave = value / self.N
        # print("ave = ", ave)
        # 求公平程度: 目标函数
        value = 0
        for row, column in li:
            value += (ave - self.wlMatrtix[row, column]) ** 2
        return value, ave

    def calculate_step1(self):
        """
        STEP1: 确定工作负荷均值
        c_av: 工作负荷均值
        c_av_min: 工作负荷均值上限
        c_av_max: 工作负荷均值下限
        :return:
        """
        self.c_av_min = self.wlMatrtix.min()
        self.c_av_max = self.wlMatrtix.max()
        self.c_av   = self.c_av_min
        self.c_fore = self.c_av_min  - 1 # 定义初始工作负荷

    def calculate_step2(self):
        """
        STEP2:确定迭代步长
        """
        li = []
        for i in range(self.N):
            for j in range(self.N):
                li.append(self.wlMatrtix[i][j])
        li = sorted(li)
        subnum = 9999999
        for i in range(self.N - 1):
            if (li[i + 1] - li[i]) < subnum:
                subnum = li[i + 1] - li[i]
        self.delta = max((subnum + 1) / self.N,step_compensation)   # 这里要+1，避免出现0

    def calculate_step3and4(self):
        """
        STEP3: 调用匈牙利算法
        STEP4: 更换新的工作负荷均值
        初始c_av:  self.c_av
        :return:
        """
        i = 0
        while self.c_av < self.c_av_max and i < N * 10:
             print("\nstart calculate ... : loop_num[", i, "]")
             c_initial = self.c_av     # 保存当前c
             # STEP3: 调用匈牙利算法
             j = 0
             while True:
                 # 根据算法（式子1.13）规则，要在这里调整工作负荷矩阵。匈牙利算法的接受值是一个标准工作负荷矩阵
                 workMatrtix = (self.wlMatrtix - self.c_av) ** 2
                 hungarian = Hungarian(workMatrtix, is_profit_matrix=False, is_FAP=True, FAP_num=self.c_av)
                 hungarian.calculate()
                 # f_r    = hungarian.get_total_potential()          # 目标函数值
                 x = hungarian.get_results()                         # 可行解
                 f_r_hat, c_ave_hat  = self.calculateCav(x)       # 计算出新的 c

                 if abs(c_ave_hat - self.c_av) > self.gamma:
                     # 未收敛：更新c_av
                     self.c_av = c_ave_hat
                 else:
                     # 收敛了就要保存结果。找机会break
                     if f_r_hat < self.result_min or self.c_av == self.c_av_min:  # 执行条件：第一次执行 or 搜索到了新的最小值
                         self.result_min = f_r_hat  # 更新最小均值
                         self.result_x = x  # 更新最小可行解
                         self.result_ave = c_ave_hat
                     # 保存本次收敛的结果，用于绘图
                     self.result_plt_y.append(c_ave_hat)  # 保存variance，这同时也是目标函数
                     self.result_plt_x.append(c_initial)

                     # STEP4：更换新的工作负荷均值
                     # 原论文在这里的描述比较模糊
                     if self.c_fore == c_ave_hat:
                         #
                         self.c_av = c_initial + self.delta
                         my_cav.append(self.c_av)
                         break
                     else:
                         # c_av 不等于 c_fore
                         self.c_fore = c_ave_hat
                         self.c_av = c_ave_hat + self.delta
                         my_cav.append(self.c_av)
                         break
                     # self.c_av = c_initial + delta
                     # break
                 if j == 10:                 # 限制迭代次数。我的电脑运算量太大了会崩，原作者没有这样的定义
                     break
                 print("\tsearch for convergent c: loop_num[", j, "]")
                 j += 1
                 print('\tc_av, c_av_hat, x: ', self.c_av, c_ave_hat, x)
                 # end while
             i += 1
             # end while
             # STEP4：更换新的工作负荷均值
             # c = c_initial + delta
         # end while

    def drawDiagram(self):
        # 创建画布
        plt.figure(figsize=(20, 8), dpi=100)

        # 绘制图像
        list1 = self.result_plt_x
        list2 = self.result_plt_y
        plt.plot(list1, list2)

        # 显示图像, 此方法会释放figure资源，之后就不能保存惹
        plt.show()

    def getResult(self,whether_draw=True):
        print("result:")
        print("最佳目标函数: ", self.result_min)
        print("此时对应解: ", self.result_x)
        print("此时对应的平均工作负荷：", self.result_ave)

        # 绘制图像结果
        if whether_draw:
            self.drawDiagram()

    def calculate(self):
        self.calculate_step1()
        self.calculate_step2()
        self.calculate_step3and4()


if __name__ == '__main__':
    fap = FAP(WorkloadMatrtix, N, step_compensation, gamma)
    fap.calculate()
    fap.getResult(whether_draw=True)

    # plt.figure(figsize=(20, 8), dpi=100)
    # plt.plot([i + 1 for i in range(len(my_cav))], my_cav)
    # plt.show()
