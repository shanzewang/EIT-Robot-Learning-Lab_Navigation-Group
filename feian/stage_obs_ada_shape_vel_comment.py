# -*- coding: utf-8 -*-
"""
机器人导航环境模拟器
====================

这个文件定义了基于ROS的机器人导航环境，用于强化学习训练。
主要功能包括：激光雷达数据处理、机器人位置控制、目标点生成、碰撞检测等。

依赖关系：
- ROS (Robot Operating System) - 机器人中间件
- Stage模拟器 - 2D机器人模拟环境
- OpenCV - 图像处理库
- TensorFlow - 深度学习框架

调用关系：
- 被 SPN_ours_1e6.py 中的主训练循环调用
- 为强化学习算法提供环境交互接口
"""

import rospy  # ROS Python库
import math
import time
import numpy as np
import cv2  # OpenCV图像处理库
import copy
import tf  # ROS的坐标变换库
import random
from collections import deque  # 双端队列，用于存储历史动作
from scipy.stats import truncnorm  # 截断正态分布

# ROS消息类型导入
import std_srvs.srv
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, Point, Pose, Pose2D
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from actionlib_msgs.msg import GoalID
from std_msgs.msg import Int8


class StageWorld:
    """
    机器人导航环境类
    ================
    
    这个类封装了完整的机器人导航环境，包括：
    1. 传感器数据处理（激光雷达、里程计）
    2. 机器人运动控制
    3. 目标点生成和管理
    4. 碰撞检测和安全监控
    5. 奖励函数计算
    6. 环境状态更新
    
    参数说明：
    beam_num: 激光雷达光束数量，用于降采样处理
    """
    
    def __init__(self, beam_num):
        """
        初始化机器人环境
        
        Args:
            beam_num (int): 激光雷达光束数量（通常为540）
        """
        # 初始化ROS节点
        rospy.init_node('StageWorld', anonymous=False)
        
        # ============= 加载测试数据集 =============
        # 加载预定义的目标点和初始位置集合（用于测试）
        goal_set1 = np.load('goal_set' + str(115) + '.npy')      # 目标点坐标集合
        robot_set1 = np.load('robot_set' + str(115) + '.npy')    # 机器人初始位置集合
        goal_set = goal_set1
        robot_set = robot_set1
        self.test_targets = goal_set     # 测试用目标点
        self.test_initials = robot_set   # 测试用初始位置

        config_set1 = np.load('config_set_nev1.npy') 
        self.config_initials = config_set1

        # ============= 基础参数初始化 =============
        self.move_base_goal = PoseStamped()        # 移动目标位姿
        self.image_size = [224, 224]               # 图像尺寸（如果使用视觉传感器）
        self.bridge = CvBridge()                   # ROS图像桥接器

        # 机器人状态相关变量
        self.object_state = [0, 0, 0, 0]          # 物体状态
        self.object_name = []                      # 物体名称列表
        self.stalled = False                       # 机器人是否卡住标志
        self.crash_stop = False                    # 碰撞停止标志

        # 机器人速度控制
        self.self_speed = [0.3, 0.0]              # [线速度, 角速度]
        self.default_states = None                 # 默认状态
        
        # 历史动作缓存（用于动作平滑）
        self.past_actions = deque(maxlen=2)        # 存储最近2个动作
        for initial_zero in range(2):
            self.past_actions.append(0)           # 初始化为0

        # 时间和步数控制
        self.start_time = time.time()              # 训练开始时间
        self.max_steps = 10000                     # 最大步数
        self.gap = 0.5                            # 间隔参数

        # ============= 传感器数据相关 =============
        self.scan = None                          # 激光雷达数据
        self.beam_num = beam_num                  # 激光束数量
        self.laser_cb_num = 0                     # 激光回调次数计数器
        self.rot_counter = 0                      # 旋转计数器

        # ============= 环境和相位控制 =============
        self.now_phase = 1                        # 当前阶段
        self.next_phase = 4                       # 下一阶段
        self.step_target = [0.0, 0.0]            # 步骤目标
        self.step_r_cnt = 0.0                     # 步骤奖励计数
        self.stop_counter = 0                     # 停止计数器

        # ============= 动作空间定义 =============
        # 定义动作的最大值和最小值 [线速度, 角速度]
        self.max_action = [0.7, np.pi / 2]       # 最大动作值
        self.min_action = [0.0, -np.pi / 2]      # 最小动作值
        self.ratio = 1.0                          # 速度比例因子

        # 初始化机器人速度
        self.self_speed = [0.3 / self.ratio, 0.0]
        self.target_point = [0, 5.5]             # 目标点坐标

        # ============= 地图处理 =============
        # 加载环境地图
        map_img = cv2.imread('Obstacles3.jpg', 0)  # 以灰度模式读取地图
        ret, binary_map = cv2.threshold(map_img, 10, 1, cv2.THRESH_BINARY)  # 二值化处理
        binary_map = 1 - binary_map               # 反转：0表示障碍物，1表示自由空间
        self.map = binary_map.astype(np.float32)  # 转换为浮点型
        
        # 获取地图尺寸
        height, width = binary_map.shape
        self.map_pixel = np.array([width, height])  # 地图像素尺寸

        # ============= 多尺度地图配置 =============
        # 定义7个不同尺度的地图（用于课程学习）
        self.map_sizes = np.zeros((7, 2))
        for map_no in range(7):
            # 每个地图尺寸按0.8的比例递减
            self.map_sizes[map_no, 0] = 15.0 * (0.8**map_no)  # x方向尺寸
            self.map_sizes[map_no, 1] = 15.0 * (0.8**map_no)  # y方向尺寸
        
        self.map_origin = self.map_pixel / 2 - 1   # 地图原点（像素坐标）

        # ============= 机器人物理参数 =============
        self.robot_size = 0.4                     # 机器人尺寸
        self.target_size = 0.4                    # 目标点尺寸
        
        # 机器人碰撞检测范围（考虑机器人形状为矩形）
        self.robot_range_x1 = 0.4                 # 前方检测范围
        self.robot_range_x2 = 0.4                 # 后方检测范围
        self.robot_range = 0.2                    # 基础检测范围
        self.robot_range_y = 0.4                  # 侧方检测范围
        
        # 运动约束参数
        self.max_acc = [2.0, 2.0]                 # 最大加速度 [线性, 角度]

        # ============= 地图中心点配置 =============
        # 定义不同环境的中心点坐标
        self.map_center = np.zeros((9, 2))
        for map_no in range(7):
            if map_no == 0:
                self.map_center[map_no, 0] = -20   # 环境0的x坐标
                self.map_center[map_no, 1] = 0     # 环境0的y坐标
            elif map_no == 1:
                self.map_center[map_no, 0] = -6
                self.map_center[map_no, 1] = 0
            elif map_no == 2:
                self.map_center[map_no, 0] = 5
                self.map_center[map_no, 1] = 0
            elif map_no == 3:
                self.map_center[map_no, 0] = 14
                self.map_center[map_no, 1] = 0
            elif map_no == 4:
                self.map_center[map_no, 0] = 21
                self.map_center[map_no, 1] = 0
            elif map_no == 5:
                self.map_center[map_no, 0] = 26.5
                self.map_center[map_no, 1] = 0
            elif map_no == 6:
                self.map_center[map_no, 0] = 31
                self.map_center[map_no, 1] = 0

        # ============= 地图渲染参数 =============
        self.robot_value = 0.33                   # 机器人在地图上的像素值
        self.target_value = 0.66                  # 目标点在地图上的像素值
        self.path_value = 0.1                     # 路径在地图上的像素值
        
        self.env = 0                              # 当前环境编号
        self.control_period = 0.2                 # 控制周期（秒）

        # ============= ROS发布者和订阅者 =============
        # 发布者：发送控制命令
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=100)         # 速度控制发布者
        self.pose_publisher = rospy.Publisher('cmd_pose', Pose2D, queue_size=1000)  # 位姿发布者
        rospy.loginfo("Publisher Created: /cmd_pose")

        # 订阅者：接收传感器数据
        self.object_state_sub = rospy.Subscriber(
            'base_pose_ground_truth', Odometry, self.GroundTruthCallBack
        )  # 真实位姿订阅者
        
        self.laser_sub = rospy.Subscriber(
            'base_scan', LaserScan, self.LaserScanCallBack
        )  # 激光雷达数据订阅者
        
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.OdometryCallBack)  # 里程计订阅者
        self.sim_clock = rospy.Subscriber('clock', Clock, self.SimClockCallBack)   # 仿真时钟订阅者
        
        # 服务客户端
        self.ResetStage = rospy.ServiceProxy('reset_positions', std_srvs.srv.Empty)  # 重置位置服务
        
        # 机器人卡住状态订阅者
        self.stalls = rospy.Subscriber("/stalled", Int8, self.update_robot_stall_data)

        # ============= 等待初始化完成 =============
        # 等待第一次激光雷达数据回调
        while self.scan is None:
            pass
        rospy.sleep(1.0)
        
        # 设置关闭时的回调函数
        rospy.on_shutdown(self.shutdown)

    def GroundTruthCallBack(self, GT_odometry):
        """
        真实位姿回调函数
        
        接收来自仿真器的真实位姿数据（用于训练和评估）
        包括位置(x,y)、方向角(theta)和速度信息
        
        Args:
            GT_odometry (Odometry): 包含位置、方向和速度的里程计消息
        """
        # 提取四元数表示的方向
        Quaternions = GT_odometry.pose.pose.orientation
        
        # 将四元数转换为欧拉角
        Euler = tf.transformations.euler_from_quaternion(
            [Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w]
        )
        
        # 存储真实状态 [x坐标, y坐标, 偏航角]
        self.state_GT = [
            GT_odometry.pose.pose.position.x,
            GT_odometry.pose.pose.position.y,
            Euler[2],  # 偏航角（绕z轴旋转）
        ]
        
        # 计算线速度大小
        v_x = GT_odometry.twist.twist.linear.x   # x方向线速度
        v_y = GT_odometry.twist.twist.linear.y   # y方向线速度
        v = np.sqrt(v_x**2 + v_y**2)             # 合成线速度
        
        # 存储真实速度 [线速度, 角速度]
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]

    def update_robot_stall_data(self, msg):
        """
        更新机器人卡住状态
        
        Args:
            msg (Int8): 卡住状态消息（0表示正常，1表示卡住）
        """
        self.stalled = msg.data

    def ImageCallBack(self, img):
        """
        图像数据回调函数（如果使用视觉传感器）
        
        Args:
            img: 图像消息
        """
        self.image = img

    def set_robot_pose(self):
        """
        随机设置机器人位置
        
        在当前环境的有效区域内随机生成机器人位置，
        确保位置不与障碍物重叠
        """
        robot_pose_data = Pose2D()
        
        # 在地图范围内随机生成x和y坐标
        x = random.uniform(
            -(self.map_size[0] / 2 - self.target_size),
            self.map_size[0] / 2 - self.target_size,
        )
        y = random.uniform(
            -(self.map_size[1] / 2 - self.target_size),
            self.map_size[1] / 2 - self.target_size,
        )
        
        # 检查生成的位置是否有效（不与障碍物重叠）
        while not self.robotPointCheck(x, y) and not rospy.is_shutdown():
            x = random.uniform(
                -(self.map_size[0] / 2 - self.target_size),
                self.map_size[0] / 2 - self.target_size,
            )
            y = random.uniform(
                -(self.map_size[1] / 2 - self.target_size),
                self.map_size[1] / 2 - self.target_size,
            )
        
        # 转换到全局坐标系并发布
        robot_pose_data.x = x + self.map_center[self.env, 0]
        robot_pose_data.y = y + self.map_center[self.env, 1]
        self.pose_publisher.publish(robot_pose_data)

    def targetPointCheck(self, x, y):
        """
        检查目标点位置是否有效
        
        Args:
            x (float): 目标点x坐标
            y (float): 目标点y坐标
            
        Returns:
            bool: True表示位置有效，False表示与障碍物重叠
        """
        target_x = x
        target_y = y
        pass_flag = True
        
        # 转换为像素坐标
        x_pixel = int(target_x * self.R2P[0] + self.map_origin[0])
        y_pixel = int(target_y * self.R2P[1] + self.map_origin[1])
        
        # 计算检查窗口大小
        window_size = int(self.target_size * np.amax(self.R2P))
        
        # 检查目标点周围的区域是否有障碍物
        for x in range(
            np.amax([0, x_pixel - window_size]),
            np.amin([self.map_pixel[0] - 1, x_pixel + window_size]),
        ):
            for y in range(
                np.amax([0, y_pixel - window_size]),
                np.amin([self.map_pixel[1] - 1, y_pixel + window_size]),
            ):
                # 检查地图上的像素值（1表示障碍物）
                if self.map[self.map_pixel[1] - y - 1, x] == 1:
                    pass_flag = False
                    break
            if not pass_flag:
                break
        
        return pass_flag

    def robotPointCheck(self, x, y):
        """
        检查机器人位置是否有效
        
        考虑机器人的实际尺寸（矩形形状），检查是否与障碍物碰撞
        
        Args:
            x (float): 机器人x坐标
            y (float): 机器人y坐标
            
        Returns:
            bool: True表示位置有效，False表示会发生碰撞
        """
        target_x = x
        target_y = y
        pass_flag = True
        
        # 转换为像素坐标
        x_pixel = int(target_x * self.R2P[0] + self.map_origin[0])
        y_pixel = int(target_y * self.R2P[1] + self.map_origin[1])
        
        # 计算机器人各方向的检查窗口大小
        window_size_x1 = int(self.robot_range_x1 * np.amax(self.R2P))  # 前方
        window_size_x2 = int(self.robot_range_x2 * np.amax(self.R2P))  # 后方
        window_size_y = int(self.robot_range_y * np.amax(self.R2P))    # 侧方
        
        # 检查机器人占用区域是否有障碍物
        for x in range(
            np.amax([0, x_pixel - window_size_x2]),
            np.amin([self.map_pixel[0] - 1, x_pixel + window_size_x1]),
        ):
            for y in range(
                np.amax([0, y_pixel - window_size_y]),
                np.amin([self.map_pixel[1] - 1, y_pixel + window_size_y]),
            ):
                if self.map[self.map_pixel[1] - y - 1, x] == 1:
                    pass_flag = False
                    break
            if not pass_flag:
                break
        
        return pass_flag

    def LaserScanCallBack(self, scan):
        """
        激光雷达数据回调函数
        
        接收并处理激光雷达扫描数据，这是环境感知的主要传感器
        
        Args:
            scan (LaserScan): 激光雷达扫描消息
        """
        # 存储激光雷达参数
        self.scan_param = [
            scan.angle_min,        # 最小扫描角度
            scan.angle_max,        # 最大扫描角度
            scan.angle_increment,  # 角度增量
            scan.time_increment,   # 时间增量
            scan.scan_time,        # 扫描时间
            scan.range_min,        # 最小测距范围
            scan.range_max,        # 最大测距范围
        ]
        
        # 将激光雷达距离数据转换为numpy数组
        self.scan = np.array(scan.ranges)
        self.laser_cb_num += 1  # 回调次数计数

    def OdometryCallBack(self, odometry):
        """
        里程计数据回调函数
        
        接收机器人的估计位姿和速度信息（相对于真实值可能有噪声）
        
        Args:
            odometry (Odometry): 里程计消息
        """
        # 提取四元数并转换为欧拉角
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion(
            [Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w]
        )
        
        # 存储估计状态 [x坐标, y坐标, 偏航角]
        self.state = [
            odometry.pose.pose.position.x,
            odometry.pose.pose.position.y,
            Euler[2],
        ]
        
        # 存储估计速度 [线速度, 角速度]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def SimClockCallBack(self, clock):
        """
        仿真时钟回调函数
        
        Args:
            clock (Clock): 仿真时钟消息
        """
        # 将仿真时间转换为浮点数（秒）
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.0

    def GetImageObservation(self):
        """
        获取图像观测（如果使用视觉传感器）
        
        将ROS图像消息转换为OpenCV格式并调整大小
        
        Returns:
            numpy.ndarray: 处理后的图像数据
        """
        try:
            # ROS图像转换为OpenCV图像
            cv_img = self.bridge.imgmsg_to_cv2(self.image, "bgr8")
        except Exception as e:
            raise e
        
        # 调整图像大小
        dim = (self.image_size[0], self.image_size[1])
        cv_resized_img = cv2.resize(cv_img, dim, interpolation=cv2.INTER_AREA)
        
        try:
            # 转换回ROS图像消息并发布
            resized_img = self.bridge.cv2_to_imgmsg(cv_resized_img, "bgr8")
        except Exception as e:
            raise e
        
        self.resized_ob.publish(resized_img)
        return cv_resized_img

    def GetLaserObservation(self):
        """
        获取降采样的激光雷达观测
        
        将原始激光雷达数据降采样为指定数量的光束，用于减少计算量
        
        Returns:
            numpy.ndarray: 降采样后的激光雷达数据
        """
        scan = copy.deepcopy(self.scan)
        
        # 处理无效数据：将NaN和无限大值设为10.0米
        scan[np.isnan(scan)] = 10.0
        scan[np.isinf(scan)] = 10.0
        
        raw_beam_num = len(scan)           # 原始光束数量
        sparse_beam_num = self.beam_num    # 目标光束数量
        step = float(raw_beam_num) / sparse_beam_num  # 采样步长
        
        # 采样右半部分
        sparse_scan_right = []
        index = 0.0
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index += step
        
        # 采样左半部分
        sparse_scan_left = []
        index = raw_beam_num - 1.0
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index -= step
        
        # 合并左右两部分
        scan_sparse = np.concatenate(
            (sparse_scan_right, sparse_scan_left[::-1]), axis=0
        )
        return scan_sparse

    def GetNoisyLaserObservation(self):
        """
        获取带噪声的激光雷达观测
        
        向激光雷达数据添加噪声，模拟真实传感器的不确定性
        
        Returns:
            numpy.ndarray: 带噪声的激光雷达数据
        """
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 2.0  # 处理NaN值
        
        # 添加均匀噪声
        nuniform_noise = np.random.uniform(-0.01, 0.01, scan.shape)
        
        # 添加与距离成比例的高斯噪声
        linear_noise = np.multiply(np.random.normal(0.0, 0.01, scan.shape), scan)
        
        # 合成噪声
        noise = nuniform_noise + linear_noise
        scan += noise
        
        # 确保距离非负
        scan[scan < 0.0] = 0.0
        
        return scan

    def GetSelfState(self):
        """获取机器人估计状态"""
        return self.state

    def GetSelfStateGT(self):
        """获取机器人真实状态"""
        return self.state_GT

    def GetSelfSpeedGT(self):
        """获取机器人真实速度"""
        return self.speed_GT

    def GetSelfSpeed(self):
        """获取机器人估计速度"""
        return self.speed

    def GetSimTime(self):
        """获取仿真时间"""
        return self.sim_time

    def ResetWorld(self, env_no, length1, length2, width):
        """
        重置环境世界（完整重置）
        
        重置机器人位置、环境参数和训练相关变量
        
        Args:
            env_no (int): 环境编号（0-6，对应不同难度）
            length1 (float): 机器人长度1（前方）
            length2 (float): 机器人长度2（后方）  
            width (float): 机器人宽度
            
        Returns:
            float: 最大线速度动作值
        """
        rospy.sleep(4.0)  # 等待环境稳定
        
        # 重置历史动作缓存
        self.past_actions = deque(maxlen=2)
        for initial_zero in range(2):
            self.past_actions.append(0)
        
        # 重置动作和加速度限制
        self.max_action[0] = 2.0      # 最大线速度
        self.max_action[1] = np.pi    # 最大角速度
        self.max_acc[0] = 2.5         # 最大线加速度
        self.max_acc[1] = 3.2         # 最大角加速度
        # print("action bound is", self.max_action)
        
        # 保存机器人几何参数
        self.length1 = length1
        self.length2 = length2
        self.width = width
        
        # 更新碰撞检测范围（考虑机器人尺寸）
        self.robot_range_x1 = self.length1 + 0.15  # 前方检测范围
        self.robot_range_x2 = self.length2 + 0.15  # 后方检测范围
        self.robot_range_y = self.width + 0.15     # 侧方检测范围
        
        # 重置计数器和状态标志
        self.stop_counter = 0.0
        self.crash_stop = False
        
        # 设置当前环境
        self.env = env_no
        self.map_size = self.map_sizes[env_no]  # 获取对应环境的地图尺寸
        
        # 根据环境难度设置目标点大小
        if env_no < 2:
            self.target_size = 0.6      # 简单环境：较大目标
        elif env_no < 5:
            self.target_size = 0.4      # 中等环境：中等目标
        else:
            self.target_size = 0.3      # 困难环境：较小目标
        
        # 计算真实坐标到像素坐标的转换比例
        self.R2P = self.map_pixel / self.map_size
        
        # 随机设置机器人初始位置
        self.set_robot_pose()
        
        # 重置运动参数
        self.stalls                    # 卡住状态
        self.self_speed = [0.0, 0.0]   # 初始速度为0
        self.step_target = [0.0, 0.0]  # 步骤目标
        self.step_r_cnt = 0.0          # 步骤奖励计数
        self.ratio = 1.0               # 速度比例
        self.start_time = time.time()   # 重置开始时间
        
        rospy.sleep(3.0)  # 等待设置生效
        return self.max_action[0]

    def Reset(self, env_no):
        """
        重置环境（简化版本，不改变机器人几何参数）
        
        Args:
            env_no (int): 环境编号
            
        Returns:
            float: 最大线速度动作值
        """
        rospy.sleep(4.0)
        
        # 重置历史动作
        self.past_actions = deque(maxlen=2)
        for initial_zero in range(2):
            self.past_actions.append(0)
        
        # 重置状态
        self.stop_counter = 0.0
        self.crash_stop = False
        self.env = env_no
        self.map_size = self.map_sizes[env_no]
        
        # 设置目标大小
        if env_no < 2:
            self.target_size = 0.6
        else:
            self.target_size = 0.4
        
        self.R2P = self.map_pixel / self.map_size
        
        # 重置运动参数
        self.stalls
        self.self_speed = [0.0, 0.0]
        self.step_target = [0.0, 0.0]
        self.step_r_cnt = 0.0
        self.ratio = 1.0
        self.start_time = time.time()
        
        rospy.sleep(3.0)
        return self.max_action[0]

    def Control(self, action):
        """
        执行控制动作
        
        根据动作指令控制机器人运动，考虑加速度限制和动作约束
        
        Args:
            action (list): 动作指令 [线速度比例, 角速度比例]，范围[-1, 1]
        """
        # 获取当前速度
        [v, w] = self.GetSelfSpeed()
        
        # 计算目标线速度，考虑加速度限制
        self.self_speed[0] = np.clip(
            action[0] * self.max_action[0],                    # 目标速度
            v - self.max_acc[0] * self.control_period,         # 最小允许速度（减速限制）
            v + self.max_acc[0] * self.control_period,         # 最大允许速度（加速限制）
        )
        
        # 计算目标角速度，考虑角加速度限制
        self.self_speed[1] = np.clip(
            action[1] * self.max_action[1],                    # 目标角速度
            w - self.max_acc[1] * self.control_period,         # 最小允许角速度
            w + self.max_acc[1] * self.control_period,         # 最大允许角速度
        )
        
        # print(v)  # 打印当前线速度用于调试
        
        # 构造ROS速度消息
        move_cmd = Twist()
        move_cmd.linear.x = self.self_speed[0]   # 前进速度
        move_cmd.linear.y = 0.0                  # 侧向速度（差分驱动机器人为0）
        move_cmd.linear.z = 0.0                  # 垂直速度（地面机器人为0）
        move_cmd.angular.x = 0.0                 # 绕x轴角速度（地面机器人为0）
        move_cmd.angular.y = 0.0                 # 绕y轴角速度（地面机器人为0）
        move_cmd.angular.z = self.self_speed[1]  # 绕z轴角速度（偏航角速度）
        
        # 发布速度指令
        self.cmd_vel.publish(move_cmd)

    def set_robot_pose_test(self,i,env_no,robot_no):
        self.max_action[0] = self.config_initials[robot_no,i,3]
        self.max_action[1] = self.config_initials[robot_no,i,4]*2
        # change the range of the maximum angular velocity from 180 degree/s to 360 degree/s
        self.max_acc[0] = self.config_initials[robot_no,i,5]
        self.max_acc[1] = self.config_initials[robot_no,i,6]
        #print("action bound is", self.max_action,"acc bound is", self.max_acc)
        self.length1=self.config_initials[robot_no,i,0]
        self.length2=self.config_initials[robot_no,i,1]
        self.width=self.config_initials[robot_no,i,2]
        self.stop_counter = 0.0
        self.crash_stop = False
        self.env=env_no
        self.map_size = self.map_sizes[env_no]
#        self.set_robot_pose()
        self.stalls
        robot_pose_data = Pose2D()
        #print(self.length1,self.length2,self.width)
        x = self.test_initials[robot_no,i,0]*(1.25**(6-env_no))+self.map_center[self.env,0]
        y = self.test_initials[robot_no,i,1]*(1.25**(6-env_no))+self.map_center[self.env,1]
        #print(x,y)
#        while not self.robotPointCheck(x,y) and not rospy.is_shutdown():
#            x = random.uniform(-(self.map_size[0]/2 - self.target_size), self.map_size[0]/2 - self.target_size)
#            y = random.uniform(-(self.map_size[1]/2 - self.target_size), self.map_size[1]/2 - self.target_size)
        robot_pose_data.theta = 0
        robot_pose_data.x = x
        robot_pose_data.y = y
        self.pose_publisher.publish(robot_pose_data)
#        print([x,y])
        rospy.sleep(2.)
        return self.max_action[0]
    
    def GenerateTargetPoint_test(self,i,env_no,robot_no):
        self.env=env_no

        self.target_point = self.test_targets[robot_no,i,:]*(1.25**(6-env_no))+self.map_center[self.env,:]
        x = self.target_point[0]
        y = self.target_point[1]
        #print(x,y)
        self.pre_distance = np.sqrt(x**2 + y**2)
        self.distance = copy.deepcopy(self.pre_distance)
#        print([x,y])


    # def set_robot_pose_test(self, i):
    #     """
    #     设置测试用的机器人位置
        
    #     使用预定义的测试位置集合，用于一致性测试
        
    #     Args:
    #         i (int): 测试位置索引
    #     """
    #     robot_pose_data = Pose2D()
        
    #     # 从测试数据集获取位置
    #     x = self.test_initials[i, 0]
    #     y = self.test_initials[i, 1]
        
    #     # 设置位姿
    #     robot_pose_data.theta = 0  # 初始朝向为0
    #     robot_pose_data.x = x
    #     robot_pose_data.y = y
        
    #     # 发布位姿并等待生效
    #     self.pose_publisher.publish(robot_pose_data)
    #     rospy.sleep(2.0)

    # def GenerateTargetPoint_test(self, i):
    #     """
    #     生成测试用目标点
        
    #     使用预定义的测试目标点，确保测试的一致性和可重复性
        
    #     Args:
    #         i (int): 目标点索引
    #     """
    #     # 从测试数据集获取目标点
    #     self.target_point = self.test_targets[i, :]
    #     x = self.target_point[0]
    #     y = self.target_point[1]
        
    #     # 计算初始距离
    #     self.pre_distance = np.sqrt(x**2 + y**2)
    #     self.distance = copy.deepcopy(self.pre_distance)

    def shutdown(self):
        """
        关闭函数
        
        在节点关闭时停止机器人运动
        """
        rospy.loginfo("Stop Moving")
        self.cmd_vel.publish(Twist())  # 发送零速度指令
        rospy.sleep(1)

    def goal_to_robot(self, goal_pose, robot_pose):
        """
        将目标点坐标转换到机器人坐标系
        
        计算目标点在机器人局部坐标系中的相对位置
        
        Args:
            goal_pose (list): 目标点全局坐标 [x, y]
            robot_pose (list): 机器人全局位姿 [x, y, theta]
            
        Returns:
            list: 目标点在机器人坐标系中的表示 [y_normalized, x_normalized, distance]
        """
        theta = robot_pose[2]  # 机器人朝向角
        
        # 计算相对位置
        abs_x = goal_pose[0] - robot_pose[0]  # 全局x方向差值
        abs_y = goal_pose[1] - robot_pose[1]  # 全局y方向差值
        
        # 坐标变换矩阵（全局坐标到机器人坐标）
        trans_matrix = np.matrix(
            [[np.cos(theta), np.sin(theta)], 
             [-np.sin(theta), np.cos(theta)]]
        )
        
        # 应用坐标变换
        rela = np.matmul(trans_matrix, np.array([[abs_x], [abs_y]]))
        rela_x = rela[0, 0]  # 机器人坐标系x方向（前方）
        rela_y = rela[1, 0]  # 机器人坐标系y方向（左侧）
        
        # 计算距离和角度
        rela_distance = np.maximum(np.sqrt(rela_x**2 + rela_y**2), 1e-8)
        rela_angle = np.arctan2(rela_y, rela_x)
        
        # 归一化表示
        target_pose = [rela_y / rela_distance, rela_x / rela_distance, rela_distance]
        
        return target_pose

    def step(self):
        """
        环境步进函数
        
        这是强化学习环境的核心函数，执行一步环境更新并返回观测、奖励等信息
        
        Returns:
            tuple: (state, reward, terminate, reset, distance, robot_pose)
                - state: 环境观测状态
                - reward: 即时奖励
                - terminate: 是否终止当前episode
                - reset: 重置类型（0:失败, 1:成功）
                - distance: 到目标点的距离
                - robot_pose: 机器人当前位姿
        """
        terminate = False  # 终止标志
        self.stop_counter = 0  # 重置停止计数器
        reset = 0  # 重置类型

        # ============= 获取激光雷达观测 =============
        state = self.GetNoisyLaserObservation()  # 获取带噪声的激光雷达数据
        laser_min = np.amin(state)               # 最小距离值
        state = np.reshape(state, (540))         # 重新整形为720维向量

        # ============= 处理激光雷达数据 =============
        # 将540个激光束分组为90个方向，每组6个光束
        pool_state = np.zeros((90, 6))  # 每个方向存储6个特征
        
        for i in range(90):
            # 计算方向角度（以机器人前方为0度，逆时针）
            pool_state[i, 0] = np.cos(i * np.pi / 45.0 - np.pi / 2)  # x方向分量
            pool_state[i, 1] = np.sin(i * np.pi / 45.0 - np.pi / 2)  # y方向分量
            
            # 取8个光束中的最小距离（最保守的障碍物检测）
            dis = np.min(state[6 * i : (6 * i + 6)])
            x_dis = pool_state[i, 0] * dis  # 障碍物在x方向的距离
            y_dis = pool_state[i, 1] * dis  # 障碍物在y方向的距离
            
            pool_state[i, 2] = dis          # 距离
            pool_state[i, 3] = self.length1 # 机器人前方长度
            pool_state[i, 4] = self.length2 # 机器人后方长度
            pool_state[i, 5] = self.width   # 机器人宽度
            
            # ============= 碰撞检测 =============
            # 检查是否在机器人的碰撞区域内
            if (abs(x_dis) <= self.width and 
                y_dis <= self.length1 and 
                y_dis >= -self.length2):
                self.stop_counter += 1.0  # 增加碰撞计数

        # 将激光雷达特征重新整形为540维向量（90*6）
        pool_state = np.reshape(pool_state, (540))

        # ============= 获取机器人状态 =============
        [x, y, theta] = self.GetSelfStateGT()  # 真实位姿
        [v, w] = self.GetSelfSpeedGT()         # 真实速度

        # ============= 计算目标相对位置 =============
        self.pre_distance = copy.deepcopy(self.distance)  # 保存上一步距离
        
        # 计算目标点相对位置
        abs_x = (self.target_point[0] - x) * self.ratio
        abs_y = (self.target_point[1] - y) * self.ratio
        
        # 坐标变换到机器人坐标系
        trans_matrix = np.matrix(
            [[np.cos(theta), np.sin(theta)], 
             [-np.sin(theta), np.cos(theta)]]
        )
        rela = np.matmul(trans_matrix, np.array([[abs_x], [abs_y]]))
        rela_x = rela[0, 0]
        rela_y = rela[1, 0]
        rela_distance = np.sqrt(rela_x**2 + rela_y**2)
        
        # 更新当前距离
        [x, y, theta] = self.GetSelfStateGT()
        [v, w] = self.GetSelfSpeed()
        self.distance = rela_distance
        rela_angle = np.arctan2(rela_y, rela_x)
        
        # ============= 构造目标信息 =============
        target_pose = [rela_distance, rela_angle]  # [距离, 角度]
        cur_act = [v * self.ratio, w]              # 当前动作
        target = target_pose

        # ============= 构造完整状态向量 =============
        state = np.concatenate(
            [
                pool_state,                                    # 激光雷达特征(540维)
                target_pose,                                   # 目标位置(2维)
                cur_act,                                       # 当前速度(2维)
                [self.max_action[0], self.max_action[1],      # 动作限制(4维)
                 self.max_acc[0], self.max_acc[1]],
            ],
            axis=0,
        )

        # ============= 计算奖励 =============
        # 基础奖励：接近目标获得正奖励，远离目标获得负奖励
        reward = 2 * (self.pre_distance - self.distance)

        result = 0

        # ============= 检查卡住状态 =============
        if self.stalled:
            self.stop_counter += 1.0

        # ============= 终止条件判断 =============
        if self.stop_counter >= 1.0:
            # 碰撞或卡住：给予负奖励并终止
            reward = -10.0
            terminate = True
            reset = 0                    # 失败重置
            self.crash_stop = True
            self.stop_counter = 0
            self.stalled = False
            # print("crashed")
        else:
            if self.distance < 0.2 and not self.stalled:
                # 成功到达目标：给予正奖励并终止
                reward = 10.0
                terminate = True
                reset = 1                # 成功重置
                self.stop_counter = 0
                self.stalled = False

        return state, reward, terminate, reset, self.distance, [x, y, theta]

    def GenerateTargetPoint(self, suc_rate):
        """
        生成训练用目标点
        
        根据成功率自适应生成目标点，实现课程学习
        成功率高时生成较远的目标，成功率低时生成较近的目标
        
        Args:
            suc_rate (float): 当前成功率（0-1）
        """
        # 根据成功率计算局部窗口大小
        # 成功率越高，允许的目标点范围越大
        local_window = np.maximum(suc_rate * self.map_size[0], 0.5)
        
        # 获取机器人当前位置（转换到环境坐标系）
        [xx, yy, theta] = self.GetSelfStateGT()
        xx = xx - self.map_center[self.env, 0]  # 转换到环境局部坐标
        yy = yy - self.map_center[self.env, 1]
        
        # 在局部窗口内随机生成目标点
        x = random.uniform(
            max(-(self.map_size[0] / 2 - self.target_size), xx - local_window),
            min((self.map_size[0] / 2 - self.target_size), xx + local_window),
        )
        y = random.uniform(
            max(-(self.map_size[1] / 2 - self.target_size), yy - local_window),
            min((self.map_size[1] / 2 - self.target_size), yy + local_window),
        )
        
        # 确保目标点位置有效（不与障碍物重叠）
        while not self.targetPointCheck(x, y) and not rospy.is_shutdown():
            x = random.uniform(
                max(-(self.map_size[0] / 2 - self.target_size), xx - local_window),
                min((self.map_size[0] / 2 - self.target_size), xx + local_window),
            )
            y = random.uniform(
                max(-(self.map_size[1] / 2 - self.target_size), yy - local_window),
                min((self.map_size[1] / 2 - self.target_size), yy + local_window),
            )
        
        # 转换到全局坐标系
        self.target_point = [
            x + self.map_center[self.env, 0],
            y + self.map_center[self.env, 1],
        ]
        
        # 计算初始距离
        self.pre_distance = np.sqrt(x**2 + y**2)
        self.distance = copy.deepcopy(self.pre_distance)

    def GetLocalTarget(self):
        """
        获取目标点在机器人坐标系中的位置
        
        Returns:
            list: [local_x, local_y] 目标点在机器人坐标系中的坐标
        """
        [x, y, theta] = self.GetSelfStateGT()
        [target_x, target_y] = self.target_point
        
        # 坐标变换：全局坐标 -> 机器人坐标
        local_x = (target_x - x) * np.cos(theta) + (target_y - y) * np.sin(theta)
        local_y = -(target_x - x) * np.sin(theta) + (target_y - y) * np.cos(theta)
        
        return [local_x, local_y]

    def TargetPointCheck(self):
        """
        检查当前目标点是否有效
        
        Returns:
            bool: True表示有效，False表示与障碍物重叠
        """
        target_x = self.target_point[0]
        target_y = self.target_point[1]
        pass_flag = True
        
        # 转换为像素坐标
        x_pixel = int(target_x * self.R2P[0] + self.map_origin[0])
        y_pixel = int(target_y * self.R2P[1] + self.map_origin[1])
        window_size = int(self.robot_range * np.amax(self.R2P))
        
        # 检查目标点周围区域
        for x in range(
            np.amax([0, x_pixel - window_size]),
            np.amin([self.map_pixel[0] - 1, x_pixel + window_size]),
        ):
            for y in range(
                np.amax([0, y_pixel - window_size]),
                np.amin([self.map_pixel[1] - 1, y_pixel + window_size]),
            ):
                if self.map[self.map_pixel[1] - y - 1, x] == 1:
                    pass_flag = False
                    break
            if not pass_flag:
                break
        
        return pass_flag

    def Global2Local(self, path, pose):
        """
        将全局路径转换为机器人局部坐标系
        
        Args:
            path (list): 全局路径点列表
            pose (list): 机器人当前位姿 [x, y, theta]
            
        Returns:
            list: 局部坐标系中的路径点
        """
        x = pose[0]
        y = pose[1]
        theta = pose[2]
        local_path = copy.deepcopy(path)
        
        # 对路径中的每个点进行坐标变换
        for t in range(0, len(path)):
            local_path[t][0] = (path[t][0] - x) * np.cos(theta) + (path[t][1] - y) * np.sin(theta)
            local_path[t][1] = -(path[t][0] - x) * np.sin(theta) + (path[t][1] - y) * np.cos(theta)
        
        return local_path

    def ResetMap(self, path):
        """
        重置地图并绘制路径
        
        Args:
            path (list): 路径点列表
            
        Returns:
            numpy.ndarray: 更新后的地图
        """
        self.map = copy.deepcopy(self.raw_map)
        target_point = path[-1]  # 路径终点作为目标点
        
        # 在地图上绘制目标点
        self.map = self.DrawPoint(
            target_point,
            self.target_size,
            self.target_value,
            self.map,
            self.map_pixel,
            self.map_origin,
            self.R2P,
        )
        return self.map

    def DrawPoint(self, point, size, value, map_img, map_pixel, map_origin, R2P):
        """
        在地图上绘制点
        
        Args:
            point (list): 点坐标 [x, y]
            size: 点的大小（可以是标量或[width, height]）
            value (float): 绘制的像素值
            map_img (numpy.ndarray): 地图图像
            map_pixel (numpy.ndarray): 地图像素尺寸
            map_origin (numpy.ndarray): 地图原点
            R2P (numpy.ndarray): 真实坐标到像素坐标的转换比例
            
        Returns:
            numpy.ndarray: 更新后的地图
        """
        # 计算绘制范围
        if not isinstance(size, np.ndarray):
            # 正方形点
            x_range = [
                np.amax([int((point[0] - size / 2) * R2P[0]) + map_origin[0], 0]),
                np.amin([int((point[0] + size / 2) * R2P[0]) + map_origin[0], map_pixel[0] - 1]),
            ]
            y_range = [
                np.amax([int((point[1] - size / 2) * R2P[1]) + map_origin[1], 0]),
                np.amin([int((point[1] + size / 2) * R2P[1]) + map_origin[1], map_pixel[1] - 1]),
            ]
        else:
            # 矩形点
            x_range = [
                np.amax([int((point[0] - size[0] / 2) * R2P[0]) + map_origin[0], 0]),
                np.amin([int((point[0] + size[0] / 2) * R2P[0]) + map_origin[0], map_pixel[0] - 1]),
            ]
            y_range = [
                np.amax([int((point[1] - size[1] / 2) * R2P[1]) + map_origin[1], 0]),
                np.amin([int((point[1] + size[1] / 2) * R2P[1]) + map_origin[1], map_pixel[1] - 1]),
            ]

        # 在指定区域绘制
        for x in range(x_range[0], x_range[1] + 1):
            for y in range(y_range[0], y_range[1] + 1):
                map_img[map_pixel[1] - y - 1, x] = value
        
        return map_img

    def DrawLine(self, point1, point2, value, map_img, map_pixel, map_origin, R2P):
        """
        在地图上绘制线段
        
        Args:
            point1, point2 (list): 线段端点坐标
            value (float): 绘制的像素值
            map_img, map_pixel, map_origin, R2P: 地图相关参数
            
        Returns:
            numpy.ndarray: 更新后的地图
        """
        # 确定起点和终点
        if point1[0] <= point2[0]:
            init_point = point1
            end_point = point2
        else:
            init_point = point2
            end_point = point1

        # 转换为地图像素坐标
        map_init_point = [
            init_point[0] * R2P[0] + map_origin[0],
            init_point[1] * R2P[1] + map_origin[1],
        ]
        map_end_point = [
            end_point[0] * R2P[0] + map_origin[0],
            end_point[1] * R2P[1] + map_origin[1],
        ]

        # 使用直线方程 y = kx + b 绘制
        if map_end_point[0] > map_init_point[0]:
            k = (map_end_point[1] - map_init_point[1]) / (map_end_point[0] - map_init_point[0])
            b = map_init_point[1] - k * map_init_point[0]
            
            if abs(k) < 1.0:
                # 斜率较小时，沿x方向采样
                x_range = [
                    np.amax([int(map_init_point[0]), 0]),
                    np.amin([int(map_end_point[0]), map_pixel[0]]),
                ]
                for x in range(x_range[0], x_range[1] + 1):
                    y = int(x * k + b)
                    if y < 0:
                        y = 0
                    elif y > map_pixel[1]:
                        y = map_pixel[1]
                    if map_img[map_pixel[1] - y - 1, x] < value:
                        map_img[map_pixel[1] - y - 1, x] = value
            else:
                # 斜率较大时，沿y方向采样
                if k > 0:
                    y_range = [
                        np.amax([int(map_init_point[1]), 0]),
                        np.amin([int(map_end_point[1]), map_pixel[1]]),
                    ]
                else:
                    y_range = [
                        np.amax([int(map_end_point[1]), 0]),
                        np.amin([int(map_init_point[1]), map_pixel[1]]),
                    ]
                for y in range(y_range[0], y_range[1] + 1):
                    x = int((y - b) / k)
                    if x < 0:
                        x = 0
                    elif x > map_pixel[0]:
                        x = map_pixel[0]
                    if map_img[map_pixel[1] - y - 1, x] < value:
                        map_img[map_pixel[1] - y - 1, x] = value
        else:
            # 垂直线的情况
            x_mid = map_end_point[0]
            x_range = [
                np.amax([int(x_mid - width / 2), 0]),
                np.amin([int(x_mid + width / 2), map_pixel[0]]),
            ]
            for x in range(x_range[0], x_range[1] + 1):
                y_range = [int(map_init_point[1]), int(map_end_point[1])]
                for y in range(y_range[0], y_range[1] + 1):
                    map_img[map_pixel[1] - y - 1, x] = value
        
        return map_img

    def RenderMap(self, path):
        """
        渲染包含机器人和目标的地图
        
        Args:
            path (list): 路径点列表
            
        Returns:
            numpy.ndarray: 渲染后的地图
        """
        [x, y, theta] = self.GetSelfStateGT()
        
        # 重置地图并绘制路径
        self.ResetMap(path)
        
        # 在地图上绘制机器人位置
        self.map = self.DrawPoint(
            [x, y],
            self.robot_size,
            self.robot_value,
            self.map,
            self.map_pixel,
            self.map_origin,
            self.R2P,
        )
        return self.map

    def PIDController(self):
        """
        PID控制器（用于初期训练或作为基准）
        
        实现简单的比例控制，使机器人朝向目标点移动
        
        Returns:
            list: 控制动作 [线速度, 角速度]
        """
        action_bound = self.max_action
        X = self.GetSelfState()  # 当前状态（未使用）
        X_t = self.GetLocalTarget() * np.array([self.ratio, self.ratio])  # 局部目标
        P = np.array([1, 10.0])  # 比例控制增益 [线速度增益, 角速度增益]
        
        # 计算控制输出
        Ut = X_t * P

        # 应用动作限制
        if Ut[0] < -action_bound[0]:
            Ut[0] = -action_bound[0]
        elif Ut[0] > action_bound[0]:
            Ut[0] = action_bound[0]

        if Ut[1] < -action_bound[1]:
            Ut[1] = -action_bound[1]
        elif Ut[1] > action_bound[1]:
            Ut[1] = action_bound[1]
        
        Ut[0] = Ut[0]  # 线速度
        Ut[1] = Ut[1]  # 角速度

        return Ut

    def OAController(self, action_bound, last_action):
        """
        障碍物避让控制器
        
        基于激光雷达数据的简单避障算法
        
        Args:
            action_bound (list): 动作边界
            last_action (list): 上一个动作
            
        Returns:
            list: 避障动作
        """
        # 获取处理后的激光雷达数据
        scan = (self.GetLaserObservation() + 0.5) * 10.0 - 0.19
        beam_num = len(scan)
        
        # 只考虑前方的激光束（中间部分）
        mid_scan = scan[int(beam_num / 4) : int(beam_num / 4) * 3]
        threshold = 1.2  # 安全距离阈值
        
        # 保持前进，但调整转向
        action = [last_action[0], 0.0]
        
        # 如果检测到近距离障碍物
        if np.amin(mid_scan) < threshold:
            if np.argmin(mid_scan) >= beam_num / 4:
                # 障碍物在右侧，向左转
                action[1] = -action_bound[1] * (threshold - np.amin(mid_scan) / threshold)
            else:
                # 障碍物在左侧，向右转
                action[1] = action_bound[1] * (threshold - np.amin(mid_scan) / threshold)

        # 应用角速度限制
        if action[1] > action_bound[1]:
            action[1] = action_bound[1]
        elif action[1] < -action_bound[1]:
            action[1] = -action_bound[1]

        return [action]


# 注释：
# 这个文件实现了一个完整的机器人导航环境，主要用于强化学习训练。
# 核心特点：
# 1. 多尺度环境支持：从简单到复杂的7个环境，实现课程学习
# 2. 自适应目标生成：根据训练进度调整目标点难度
# 3. 完整的传感器模拟：激光雷达、里程计、真实位姿
# 4. 灵活的机器人模型：支持不同尺寸的矩形机器人
# 5. 丰富的状态表示：激光雷达特征+目标信息+运动状态
# 6. 实时碰撞检测：基于激光雷达和机器人几何形状
# 7. 奖励函数设计：距离奖励+成功奖励+碰撞惩罚
#
# 这个环境类是整个强化学习系统的基础，为SAC算法提供了标准的gym-like接口。