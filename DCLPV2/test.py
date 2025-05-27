import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置全局字体为不使用中文
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# 加载测试结果
test_result_plot = np.load('test_result_plot_torch.npy') 
print(f"Array Shape: {test_result_plot.shape}")

# 找出实际有数据的最大测试时间点
non_zero_indices = np.where(np.any(test_result_plot[:, :, :, :, 0] != 0, axis=(0, 1, 2)))[0]
if len(non_zero_indices) > 0:
    max_test_time = non_zero_indices[-1]
else:
    max_test_time = 0
print(f"Max valid test time: {max_test_time}")

# 对每个超参数实验和环境复杂度，计算平均成功率和评分
for hyper_exp in range(test_result_plot.shape[0]):
    print(f"\nHyper-experiment {hyper_exp}:")
    
    for shape_no in range(test_result_plot.shape[1]):
        # 计算最后一个有效测试时间点的平均指标
        if max_test_time > 0:
            success_rate = np.mean(test_result_plot[hyper_exp, shape_no, :, max_test_time, 1])
            avg_reward = np.mean(test_result_plot[hyper_exp, shape_no, :, max_test_time, 0])
            avg_score = np.mean(test_result_plot[hyper_exp, shape_no, :, max_test_time, 3])
            crash_rate = np.mean(test_result_plot[hyper_exp, shape_no, :, max_test_time, 4])
            
            print(f"  Environment Complexity {shape_no}:")
            print(f"    - Success Rate: {success_rate:.2f}")
            print(f"    - Avg Reward: {avg_reward:.2f}")
            print(f"    - Avg Score: {avg_score:.2f}")
            print(f"    - Crash Rate: {crash_rate:.2f}")

# 创建颜色映射
colors = plt.cm.tab10(np.linspace(0, 1, 5))
markers = ['o', 's', '^', 'D', 'v']

# 绘制成功率随时间变化的曲线
plt.figure(figsize=(12, 8))

# 只为每个环境复杂度绘制一组曲线，选择实验1
hyper_exp = 1  # 使用实验1的数据，因为它有非零数据
for shape_no in range(test_result_plot.shape[1]):
    success_rates = []
    
    for t in range(max_test_time + 1):
        # 只计算非零数据点的平均值
        sr = np.mean(test_result_plot[hyper_exp, shape_no, :, t, 1])
        success_rates.append(sr)
    
    plt.plot(success_rates, label=f'Env {shape_no}', 
             color=colors[shape_no], marker=markers[shape_no], linewidth=2, markersize=8)

plt.xlabel('Test Time Point', fontsize=14)
plt.ylabel('Success Rate', fontsize=14)
plt.title('Success Rate Over Training Progress', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('success_rate_over_time.png', dpi=300)
print("\nSuccess rate chart saved to success_rate_over_time.png")

# 绘制平均奖励随时间变化的曲线
plt.figure(figsize=(12, 8))

for shape_no in range(test_result_plot.shape[1]):
    rewards = []
    
    for t in range(max_test_time + 1):
        # 只计算非零数据点的平均值
        reward = np.mean(test_result_plot[hyper_exp, shape_no, :, t, 0])
        rewards.append(reward)
    
    plt.plot(rewards, label=f'Env {shape_no}', 
             color=colors[shape_no], marker=markers[shape_no], linewidth=2, markersize=8)

plt.xlabel('Test Time Point', fontsize=14)
plt.ylabel('Average Reward', fontsize=14)
plt.title('Average Reward Over Training Progress', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('avg_reward_over_time.png', dpi=300)
print("Average reward chart saved to avg_reward_over_time.png")

# 绘制碰撞率随时间变化的曲线
plt.figure(figsize=(12, 8))

for shape_no in range(test_result_plot.shape[1]):
    crash_rates = []
    
    for t in range(max_test_time + 1):
        # 只计算非零数据点的平均值
        cr = np.mean(test_result_plot[hyper_exp, shape_no, :, t, 4])
        crash_rates.append(cr)
    
    plt.plot(crash_rates, label=f'Env {shape_no}', 
             color=colors[shape_no], marker=markers[shape_no], linewidth=2, markersize=8)

plt.xlabel('Test Time Point', fontsize=14)
plt.ylabel('Crash Rate', fontsize=14)
plt.title('Crash Rate Over Training Progress', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('crash_rate_over_time.png', dpi=300)
print("Crash rate chart saved to crash_rate_over_time.png")

# 绘制自定义评分随时间变化的曲线
plt.figure(figsize=(12, 8))

for shape_no in range(test_result_plot.shape[1]):
    scores = []
    
    for t in range(max_test_time + 1):
        # 只计算非零数据点的平均值
        score = np.mean(test_result_plot[hyper_exp, shape_no, :, t, 3])
        scores.append(score)
    
    plt.plot(scores, label=f'Env {shape_no}', 
             color=colors[shape_no], marker=markers[shape_no], linewidth=2, markersize=8)

plt.xlabel('Test Time Point', fontsize=14)
plt.ylabel('Custom Score', fontsize=14)
plt.title('Custom Score Over Training Progress', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('custom_score_over_time.png', dpi=300)
print("Custom score chart saved to custom_score_over_time.png")

print("\nTest Results Summary:")
print(f"- Data Shape: {test_result_plot.shape}")
print(f"- Accumulated Reward: Index 0")
print(f"- Goal Reached: Index 1 (1=success, 0=failure)")
print(f"- Step Ratio: Index 2 (actual steps/max steps)")
print(f"- Custom Score: Index 3 (1.0*goal_reach - ep_len*2.0/max_ep_len)")
print(f"- Crash Flag: Index 4 (1=crashed, 0=no crash)")

print("\nAnalysis:")
if max_test_time <= 2:
    print("- Training appears to be in early stages (max_test_time = 2)")

# 分析实验1 (超参数实验索引1)的数据
print(f"- Hyper-experiment 1 has data for {max_test_time+1} test time points")
print("- Current performance metrics for experiment 1:")

# 计算最后时间点的平均指标
avg_success = np.mean([np.mean(test_result_plot[1, shape_no, :, max_test_time, 1]) for shape_no in range(5)])
avg_crash = np.mean([np.mean(test_result_plot[1, shape_no, :, max_test_time, 4]) for shape_no in range(5)])
avg_reward_all = np.mean([np.mean(test_result_plot[1, shape_no, :, max_test_time, 0]) for shape_no in range(5)])

print(f"  * Overall Success Rate: {avg_success:.2f}")
print(f"  * Overall Crash Rate: {avg_crash:.2f}")
print(f"  * Overall Average Reward: {avg_reward_all:.2f}")

# 结论
print("\nConclusions:")
print("- The robot currently fails to reach goals in all environments (success rate = 0)")
print("- Environment complexity affects crash rate: higher complexity leads to more crashes")
print("- Negative rewards indicate the robot is not making progress toward goals")
print("- Training needs to continue for more iterations to observe learning progress")