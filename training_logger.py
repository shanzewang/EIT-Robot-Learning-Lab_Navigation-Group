# -*- coding: utf-8 -*-
"""
训练日志管理和可视化系统
========================

这个模块提供了完整的训练过程监控、日志管理和结果可视化功能
包括学习曲线绘制、测试结果统计、训练状态跟踪等
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import os
import json
import time
from datetime import datetime
from collections import deque
import pandas as pd
from stage_obs_comment import StageWorld

class TrainingLogger:
    """训练日志管理器"""
    
    def __init__(self, experiment_id, save_dir="training_logs"):
        """
        初始化日志管理器
        
        Args:
            experiment_id (int): 实验编号
            save_dir (str): 保存目录
        """
        self.experiment_id = experiment_id
        self.save_dir = save_dir
        
        # 创建保存目录
        self.log_dir = os.path.join(save_dir, f"experiment_{experiment_id}")
        self.plots_dir = os.path.join(self.log_dir, "plots")
        self.data_dir = os.path.join(self.log_dir, "data")
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 训练数据存储
        self.training_rewards = []          # 每个episode的奖励
        self.training_steps = []            # 每个episode的步数
        self.training_success = []          # 每个episode是否成功
        self.episode_count = 0
        self.total_steps = 0
        self.total_training_steps = 0       # 总训练步数T（参数更新次数）
        
        # 测试数据存储
        self.test_results = {}              # 存储测试结果
        self.test_history = []              # 测试历史
        
        # 学习曲线数据（滑动平均）
        self.reward_window = deque(maxlen=100)    # 最近100个episode的奖励
        self.success_window = deque(maxlen=100)   # 最近100个episode的成功率
        
        # 当前状态
        self.current_phase = "INITIALIZATION"  # INITIALIZATION, TRAINING, TESTING
        self.current_env = 0
        self.current_robot_size = [0, 0, 0]
        
        # 创建日志文件
        self.log_file = os.path.join(self.log_dir, "training.log")
        self.write_log(f"实验 {experiment_id} 开始 - {datetime.now()}\n" + "="*50)
    
    def write_log(self, message):
        """写入日志文件"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
        
        print(log_message)
    
    def set_phase(self, phase, **kwargs):
        """设置当前阶段"""
        self.current_phase = phase
        
        if phase == "TRAINING":
            self.current_env = kwargs.get('env', 0)
            self.current_robot_size = kwargs.get('robot_size', [0, 0, 0])
            
        elif phase == "TESTING":
            test_round = kwargs.get('test_round', 0)
            self.write_log(f"\n🧪 开始第 {test_round} 轮性能测试")
            self.write_log("-" * 30)
    
    def log_episode_start(self, episode, env_no, robot_size, distance, success_rate, max_steps):
        """记录episode开始信息"""
        if self.current_phase == "TRAINING":
            self.write_log(
                f"📍 训练 Episode {episode:4d} | "
                f"环境{env_no} | 距离{distance:.1f}m | "
                f"尺寸({robot_size[0]:.2f},{robot_size[1]:.2f},{robot_size[2]:.2f}) | "
                f"成功率{success_rate:.3f} | 最大步数{max_steps}"
            )
    
    def log_episode_end(self, episode, reward, steps, success, crash=False, timeout=False, update_training_steps=True):
        """记录episode结束信息"""
        if self.current_phase == "TRAINING":
            self.episode_count += 1
            self.total_steps += steps
            
            # 更新总训练步数T（仅在正式训练阶段）
            if update_training_steps:
                self.total_training_steps += steps
            
            # 存储数据
            self.training_rewards.append(reward)
            self.training_steps.append(steps)
            self.training_success.append(success)
            self.reward_window.append(reward)
            self.success_window.append(success)
            
            # 确定结束原因
            if success:
                status = "✅ 成功到达"
                icon = "🎯"
            elif crash:
                status = "💥 碰撞终止"
                icon = "⚠️"
            elif timeout:
                status = "⏰ 超时终止"
                icon = "🕐"
            else:
                status = "❓ 其他原因"
                icon = "❓"
            
            # 计算滑动平均
            avg_reward = np.mean(self.reward_window)
            avg_success = np.mean(self.success_window)
            
            self.write_log(
                f"  {icon} Episode {episode:4d} 结束 | {status} | "
                f"奖励{reward:6.2f} | 步数{steps:3d} | "
                f"总训练步数T:{self.total_training_steps:6d} | "
                f"近100ep: 平均奖励{avg_reward:6.2f}, 成功率{avg_success:.3f}"
            )
            
            # 每50个episode总结一次
            if episode % 50 == 0:
                self.write_log(f"  📊 阶段总结 - 总步数: {self.total_steps}, 总训练步数T: {self.total_training_steps}, 平均奖励: {avg_reward:.2f}")
    
    def log_crash(self, context="training"):
        """记录碰撞信息（只在必要时输出）"""
        if context == "training" and self.current_phase == "TRAINING":
            pass  # 在episode_end中已经记录了碰撞信息，这里不重复输出
        elif context == "initialization":
            self.write_log("  🔄 初始位置检查中发现碰撞，重新生成...")
    
    def start_test_round(self, test_round):
        """开始测试轮次"""
        self.current_test_round = test_round
        self.current_test_results = {}
        self.write_log(f"\n🧪 第 {test_round} 轮性能测试开始")
        self.write_log("-" * 50)
    
    def log_test_group_start(self, group_id, env_no):
        """开始测试某个机器人尺寸组"""
        self.write_log(f"  📋 测试尺寸组 {group_id} (环境 {env_no})")
    
    def log_test_group_end(self, group_id, results):
        """结束测试某个机器人尺寸组"""
        success_rate = results['success_rate']
        avg_reward = results['avg_reward']
        collision_rate = results['collision_rate']
        
        self.write_log(
            f"  ✅ 尺寸组 {group_id} 完成 | "
            f"成功率: {success_rate:.3f} | "
            f"平均奖励: {avg_reward:.2f} | "
            f"碰撞率: {collision_rate:.3f}"
        )
        
        # 存储结果
        self.current_test_results[group_id] = results
    
    def end_test_round(self, env_progression):
        """结束测试轮次"""
        overall_success = np.mean([r['success_rate'] for r in self.current_test_results.values()])
        
        self.write_log(f"  🎯 第 {self.current_test_round} 轮测试完成")
        self.write_log(f"  📈 整体成功率: {overall_success:.3f}")
        self.write_log(f"  🏗️ 环境进度: {env_progression}")
        
        # 存储测试历史
        test_data = {
            'test_round': self.current_test_round,
            'timestamp': datetime.now().isoformat(),
            'overall_success_rate': overall_success,
            'group_results': self.current_test_results,
            'env_progression': env_progression.tolist()
        }
        self.test_history.append(test_data)
        
        # 保存测试结果
        self.save_test_results()
        
        self.write_log("-" * 50)
    
    def plot_learning_curves(self):
        """绘制学习曲线"""
        if len(self.training_rewards) < 10:
            return
        
        # 设置matplotlib使用英文字体，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Experiment {self.experiment_id} Learning Curves', fontsize=16)
        
        episodes = np.arange(1, len(self.training_rewards) + 1)
        
        # 1. 奖励曲线
        axes[0, 0].plot(episodes, self.training_rewards, alpha=0.3, color='blue', label='Raw Rewards')
        if len(self.training_rewards) > 10:
            # 滑动平均
            window_size = min(50, len(self.training_rewards) // 4)
            rewards_smooth = np.convolve(self.training_rewards, 
                                       np.ones(window_size)/window_size, mode='valid')
            episodes_smooth = episodes[window_size-1:]
            axes[0, 0].plot(episodes_smooth, rewards_smooth, color='red', linewidth=2, label=f'{window_size}ep Moving Average')
        
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 成功率曲线
        success_rates = np.array(self.training_success, dtype=float)
        axes[0, 1].plot(episodes, success_rates, alpha=0.3, color='green', label='Single Success')
        if len(success_rates) > 10:
            window_size = min(50, len(success_rates) // 4)
            success_smooth = np.convolve(success_rates, 
                                       np.ones(window_size)/window_size, mode='valid')
            episodes_smooth = episodes[window_size-1:]
            axes[0, 1].plot(episodes_smooth, success_smooth, color='orange', linewidth=2, 
                          label=f'{window_size}ep Moving Average')
        
        axes[0, 1].set_title('Training Success Rate')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim(-0.1, 1.1)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Episode长度
        axes[1, 0].plot(episodes, self.training_steps, alpha=0.6, color='purple')
        axes[1, 0].set_title('Episode Length')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 测试成功率历史
        if self.test_history:
            test_rounds = [t['test_round'] for t in self.test_history]
            test_success = [t['overall_success_rate'] for t in self.test_history]
            axes[1, 1].plot(test_rounds, test_success, 'ro-', linewidth=2, markersize=6)
            axes[1, 1].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Target Success Rate (0.9)')
            axes[1, 1].set_title('Test Success Rate History')
            axes[1, 1].set_xlabel('Test Round')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].set_ylim(0, 1.1)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"learning_curves_{timestamp}.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.write_log(f"  📊 Learning curves saved: {filename}")
    
    def plot_test_results_heatmap(self):
        """绘制测试结果热力图"""
        if not self.test_history:
            return
        
        # 设置matplotlib使用英文字体，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 准备数据
        test_rounds = len(self.test_history)
        groups = 5  # 5个机器人尺寸组
        
        success_matrix = np.zeros((groups, test_rounds))
        reward_matrix = np.zeros((groups, test_rounds))
        
        for i, test_data in enumerate(self.test_history):
            for group_id, results in test_data['group_results'].items():
                success_matrix[group_id, i] = results['success_rate']
                reward_matrix[group_id, i] = results['avg_reward']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 成功率热力图
        im1 = axes[0].imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[0].set_title('Test Success Rate (by Robot Size Groups)')
        axes[0].set_xlabel('Test Round')
        axes[0].set_ylabel('Robot Size Group')
        axes[0].set_yticks(range(groups))
        axes[0].set_yticklabels([f'Group {i}' for i in range(groups)])
        plt.colorbar(im1, ax=axes[0], label='Success Rate')
        
        # 在格子中显示数值
        for i in range(groups):
            for j in range(test_rounds):
                text = axes[0].text(j, i, f'{success_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        # 平均奖励热力图
        vmax_reward = np.max(reward_matrix) if np.max(reward_matrix) > 0 else 1
        vmin_reward = np.min(reward_matrix)
        im2 = axes[1].imshow(reward_matrix, cmap='viridis', aspect='auto', 
                           vmin=vmin_reward, vmax=vmax_reward)
        axes[1].set_title('Test Average Reward (by Robot Size Groups)')
        axes[1].set_xlabel('Test Round')
        axes[1].set_ylabel('Robot Size Group')
        axes[1].set_yticks(range(groups))
        axes[1].set_yticklabels([f'Group {i}' for i in range(groups)])
        plt.colorbar(im2, ax=axes[1], label='Average Reward')
        
        # 在格子中显示数值
        for i in range(groups):
            for j in range(test_rounds):
                text = axes[1].text(j, i, f'{reward_matrix[i, j]:.1f}',
                                  ha="center", va="center", color="white", fontsize=8)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_heatmap_{timestamp}.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.write_log(f"  🔥 Test results heatmap saved: {filename}")
    
    def save_training_data(self):
        """保存训练数据"""
        training_data = {
            'experiment_id': self.experiment_id,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'total_training_steps': self.total_training_steps,  # 添加总训练步数T
            'rewards': self.training_rewards,
            'steps': self.training_steps,
            'success': self.training_success,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"training_data_{self.experiment_id}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    def save_test_results(self):
        """保存测试结果"""
        filename = f"test_results_{self.experiment_id}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.test_history, f, indent=2, ensure_ascii=False)
        
        # 同时保存为CSV格式方便分析
        if self.test_history:
            df_list = []
            for test_data in self.test_history:
                test_round = test_data['test_round']
                for group_id, results in test_data['group_results'].items():
                    row = {
                        'test_round': test_round,
                        'group_id': group_id,
                        'success_rate': results['success_rate'],
                        'avg_reward': results['avg_reward'],
                        'collision_rate': results['collision_rate'],
                        'env_level': test_data['env_progression'][group_id]
                    }
                    df_list.append(row)
            
            df = pd.DataFrame(df_list)
            csv_filename = f"test_results_{self.experiment_id}.csv"
            csv_filepath = os.path.join(self.data_dir, csv_filename)
            df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
    
    def generate_summary_report(self):
        """生成总结报告"""
        if not self.training_rewards:
            return
        
        report = f"""
实验 {self.experiment_id} 训练总结报告
{'='*50}

训练统计:
- 总Episode数: {self.episode_count}
- 总环境交互步数: {self.total_steps}
- 总训练步数T: {self.total_training_steps}
- 平均每Episode步数: {np.mean(self.training_steps):.1f}
- 最终平均奖励: {np.mean(self.reward_window):.2f}
- 最终成功率: {np.mean(self.success_window):.3f}

性能指标:
- 最高单Episode奖励: {np.max(self.training_rewards):.2f}
- 最低单Episode奖励: {np.min(self.training_rewards):.2f}
- 奖励标准差: {np.std(self.training_rewards):.2f}
- 训练成功次数: {sum(self.training_success)}

测试统计:
- 测试轮次: {len(self.test_history)}
"""

        if self.test_history:
            final_test = self.test_history[-1]
            report += f"- 最终整体成功率: {final_test['overall_success_rate']:.3f}\n"
            report += f"- 环境解锁进度: {final_test['env_progression']}\n"
        
        report += f"\n报告生成时间: {datetime.now()}\n"
        
        # 保存报告
        report_filename = f"summary_report_{self.experiment_id}.txt"
        report_filepath = os.path.join(self.log_dir, report_filename)
        
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.write_log("📋 Summary report generated")
        return report

# 使用示例和集成指南
class StageWorldLogger(StageWorld):
    """扩展StageWorld类，添加日志功能"""
    
    def __init__(self, beam_num, logger=None):
        super().__init__(beam_num)
        self.logger = logger
        self.step_context = "training"  # 当前step调用的上下文
    
    def set_step_context(self, context):
        """设置step调用的上下文"""
        self.step_context = context
    
    def step(self):
        """重写step函数，添加上下文感知的日志"""
        # 调用原始step函数
        state, reward, terminate, reset, distance, robot_pose = super().step()
        
        # 根据上下文决定是否记录碰撞
        if terminate and reset == 0:  # 碰撞终止
            if self.logger and self.step_context == "initialization":
                self.logger.log_crash("initialization")
            # 训练时的碰撞在episode_end中统一处理，这里不重复输出
        
        return state, reward, terminate, reset, distance, robot_pose