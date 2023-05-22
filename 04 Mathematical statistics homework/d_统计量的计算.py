# 导入numpy库，用于计算统计量
import numpy as np

# 定义一个函数，用于接收用户输入的一组数值，并返回统计量
def get_statistics():
  # 提示用户输入一组数值，用逗号分隔，并转换为浮点数列表
  data = input("请输入一组数值，用逗号分隔：")
  data = [float(x) for x in data.split(",")]

  # 创建一个numpy数组，用于计算统计量
  data = np.array(data)

  # 计算并打印样本方差、均值、样本标准差等统计量
  print("样本方差：", data.var(ddof=1))
  print("均值：", data.mean())
  print("样本标准差：", data.std(ddof=1))
  print("最小值：", data.min())
  print("最大值：", data.max())
  print("中位数：", np.median(data))
  print("四分位数：", np.quantile(data, [0.25, 0.5, 0.75]))

# 调用函数
get_statistics()