
# import numpy as np

# # 基本用法：读取.npy文件
# data = np.load('config_set_nev1.npy')
# print("数据形状:", data.shape)
# print("数据类型:", data.dtype)
# print("数据内容:")
# print(data)

# # 如果文件可能不存在，可以添加异常处理
# try:
#     data = np.load('your_file.npy')
#     print("文件读取成功")
#     print(f"数组形状: {data.shape}")
#     print(f"数据类型: {data.dtype}")
    
#     # 查看数组的基本信息
#     print(f"数组维度: {data.ndim}")
#     print(f"数组大小: {data.size}")
#     print(f"内存使用: {data.nbytes} bytes")
    
# except FileNotFoundError:
#     print("文件未找到，请检查文件路径")
# except Exception as e:
#     print(f"读取文件时出错: {e}")

import numpy as np

# 假设你的数组名为 data，形状为 (5, 200, 7)
# data = np.load('your_file.npy')


data = data = np.load('config_set_nev1.npy')

print(f"数组形状: {data.shape}")
print("\n" + "="*50)

# 方法1: 查看每列的最小值和最大值
print("方法1: 每列的最小值和最大值")
for i in range(data.shape[2]):  # 遍历第三个维度的7列
    col_data = data[:, :, i]  # 提取第i列的所有数据
    min_val = np.min(col_data)
    max_val = np.max(col_data)
    print(f"第{i}列: 最小值={min_val:.3f}, 最大值={max_val:.3f}, 范围={max_val-min_val:.3f}")

print("\n" + "="*50)

# 方法2: 一次性计算所有列的统计信息
print("方法2: 所有列的统计信息")
# 沿着前两个维度计算统计量，保留第三个维度
min_vals = np.min(data, axis=(0, 1))  # 每列的最小值
max_vals = np.max(data, axis=(0, 1))  # 每列的最大值
mean_vals = np.mean(data, axis=(0, 1))  # 每列的均值
std_vals = np.std(data, axis=(0, 1))   # 每列的标准差

print("列索引 |   最小值   |   最大值   |   均值    |  标准差   |   范围")
print("-" * 65)
for i in range(7):
    range_val = max_vals[i] - min_vals[i]
    print(f"  {i}    | {min_vals[i]:8.3f} | {max_vals[i]:8.3f} | {mean_vals[i]:8.3f} | {std_vals[i]:8.3f} | {range_val:8.3f}")

print("\n" + "="*50)
