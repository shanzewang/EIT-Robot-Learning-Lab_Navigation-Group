# goal range changes with the success rate
import rospy
import math
import time
import numpy as np
import cv2
import copy
import tf
import random
from collections import deque
from scipy.stats import truncnorm

import std_srvs.srv
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import PoseStamped, Point, Pose, Pose2D
from std_msgs.msg import Int8


class StageWorld:
    def __init__(self, beam_num):
        # initiliaze
        rospy.init_node('StageWorld', anonymous=False)
        goal_set1 = np.load('goal_set' + str(115) + '.npy')
        robot_set1 = np.load('robot_set' + str(115) + '.npy')
        goal_set = goal_set1
        robot_set = robot_set1
        self.test_targets = goal_set
        self.test_initials = robot_set

        # ------------Params--------------------
        self.move_base_goal = PoseStamped()
        self.image_size = [224, 224]
        self.bridge = CvBridge()

        self.object_state = [0, 0, 0, 0]
        self.object_name = []
        self.stalled = False
        self.crash_stop = False

        self.self_speed = [0.3, 0.0]
        self.default_states = None
        self.past_actions = deque(maxlen=2)
        for initial_zero in range(2):
            self.past_actions.append(0)

        self.start_time = time.time()
        self.max_steps = 10000
        self.gap = 0.5
        self.last_position = [0.0, 0.0, 0.0]  # 初始化位置缓存

        self.scan = None
        self.beam_num = beam_num
        self.laser_cb_num = 0
        #        self.robot_range_bound = 0.1
        self.rot_counter = 0

        self.now_phase = 1
        self.next_phase = 4
        self.step_target = [0.0, 0.0]
        self.step_r_cnt = 0.0
        self.stop_counter = 0
        #        self.action_table = [[-1.,-1.],[0.0,-1.],[1.,-1.],[-1.,0.],[0.,0.],[1.,0.],[1.,1.],[0.,1.],[-1.,1.]]
        self.max_action = [0.7, np.pi / 2]
        self.min_action = [0.0, -np.pi / 2]
        self.ratio = 1.0

        self.self_speed = [0.3 / self.ratio, 0.0]
        self.target_point = [0, 5.5]
        map_img = cv2.imread('Obstacles3.jpg', 0)
        ret, binary_map = cv2.threshold(map_img, 10, 1, cv2.THRESH_BINARY)
        binary_map = 1 - binary_map
        self.map = binary_map.astype(np.float32)
        # cv2.imshow('img',binary_map*255)
        # cv2.waitKey(0)
        height, width = binary_map.shape
        self.map_pixel = np.array([width, height])
        self.map_sizes = np.zeros((7, 2))
        for map_no in range(7):
            self.map_sizes[map_no, 0] = 15.5 * (0.8**map_no)
            self.map_sizes[map_no, 1] = 15.5 * (0.8**map_no)
        self.map_origin = self.map_pixel / 2 - 1
        self.robot_size = 0.4
        self.target_size = 0.4
        self.robot_range_x1 = 0.4
        self.robot_range_x2 = 0.4
        self.robot_range = 0.2
        self.robot_range_y = 0.4
        self.max_acc = [2.0, 2.0]
        self.map_center = np.zeros((9, 2))
        for map_no in range(7):
            if map_no == 0:
                self.map_center[map_no, 0] = -22
                self.map_center[map_no, 1] = 0
            if map_no == 1:
                self.map_center[map_no, 0] = -7
                self.map_center[map_no, 1] = 0
            if map_no == 2:
                self.map_center[map_no, 0] = 5
                self.map_center[map_no, 1] = 0
            if map_no == 3:
                self.map_center[map_no, 0] = 14.5
                self.map_center[map_no, 1] = 0
            if map_no == 4:
                self.map_center[map_no, 0] = 22
                self.map_center[map_no, 1] = 0
            if map_no == 5:
                self.map_center[map_no, 0] = 28
                self.map_center[map_no, 1] = 0
            if map_no == 6:
                self.map_center[map_no, 0] = 33
                self.map_center[map_no, 1] = 0
        self.robot_value = 0.33
        self.target_value = 0.66
        self.path_value = 0.1
        self.env = 0
        self.control_period = 0.2

        # -----------Publisher and Subscriber-------------
        self.cmd_vel = rospy.Publisher('/robot_0/cmd_vel', Twist, queue_size=100)
        self.pose_publisher = rospy.Publisher('/robot_0/cmd_pose', Pose2D, queue_size=1000)
        rospy.loginfo("Publisher Created: /robot_0/cmd_pose")
        
        # 创建7个roomba的Publisher，避免在set_robot_pose中重复创建
        self.roomba_publishers = []
        for i in range(7):
            roomba_topic = f'/robot_{i+1}/cmd_pose'
            try:
                roomba_publisher = rospy.Publisher(roomba_topic, Pose2D, queue_size=10)
                self.roomba_publishers.append(roomba_publisher)
                rospy.loginfo(f"Publisher Created: {roomba_topic}")
            except Exception as e:
                rospy.logwarn(f"无法创建Publisher {roomba_topic}: {e}")
                self.roomba_publishers.append(None)  # 占位符，防止索引错误

        self.object_state_sub = rospy.Subscriber(
            '/robot_0/base_pose_ground_truth', Odometry, self.GroundTruthCallBack
        )
        self.laser_sub = rospy.Subscriber(
            '/robot_0/base_scan', LaserScan, self.LaserScanCallBack
        )
        self.odom_sub = rospy.Subscriber('/robot_0/odom', Odometry, self.OdometryCallBack)
        self.sim_clock = rospy.Subscriber('/clock', Clock, self.SimClockCallBack)
        #        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        #        self.goal_cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=10)

        # -----------Service-------------------
        self.ResetStage = rospy.ServiceProxy('reset_positions', std_srvs.srv.Empty)
        self.stalls = rospy.Subscriber("/robot_0/stalled", Int8, self.update_robot_stall_data)

        # Wait until the first callback
        while self.scan is None:
            pass
        rospy.sleep(1.0)
        # What function to call when you ctrl + c
        rospy.on_shutdown(self.shutdown)

    #    def Mapswitch(self):
    #        self.env = self.env+1
    #        self.map_size = 0.8*self.map_size  # 20x20m
    #        self.map_origin = self.map_pixel/2 - 1
    #        self.R2P = self.map_pixel / self.map_size

    def GroundTruthCallBack(self, GT_odometry):
        Quaternions = GT_odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion(
            [Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w]
        )
        self.state_GT = [
            GT_odometry.pose.pose.position.x,
            GT_odometry.pose.pose.position.y,
            Euler[2],
        ]
        v_x = GT_odometry.twist.twist.linear.x
        v_y = GT_odometry.twist.twist.linear.y
        v = np.sqrt(v_x**2 + v_y**2)
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]

    def update_robot_stall_data(self, msg):
        self.stalled = msg.data

    def ImageCallBack(self, img):
        self.image = img

    def set_robot_pose(self):
        robot_pose_data = Pose2D()
        x = random.uniform(
            -(self.map_size[0] / 2 - self.target_size),
            self.map_size[0] / 2 - self.target_size,
        )
        y = random.uniform(
            -(self.map_size[1] / 2 - self.target_size),
            self.map_size[1] / 2 - self.target_size,
        )
        while not self.robotPointCheck(x, y) and not rospy.is_shutdown():
            x = random.uniform(
                -(self.map_size[0] / 2 - self.target_size),
                self.map_size[0] / 2 - self.target_size,
            )
            y = random.uniform(
                -(self.map_size[1] / 2 - self.target_size),
                self.map_size[1] / 2 - self.target_size,
            )
        robot_pose_data.x = x + self.map_center[self.env, 0]
        robot_pose_data.y = y + self.map_center[self.env, 1]
        
        # 设置主机器人位置
        try:
            self.pose_publisher.publish(robot_pose_data)
            rospy.sleep(0.5)  # 给主机器人位置更新一些时间
        except Exception as e:
            rospy.logwarn(f"设置主机器人位置失败: {e}")
        
        # 为7个roomba设置位置，沿着当前环境的对角线均匀分布
        # 获取当前环境的大小和中心
        current_map_size = self.map_sizes[self.env]
        current_map_center = self.map_center[self.env]

        # 计算环境的左上角和右下角坐标（相对于环境中心）
        top_left_x = -current_map_size[0] / 2
        top_left_y = current_map_size[1] / 2
        bottom_right_x = current_map_size[0] / 2
        bottom_right_y = -current_map_size[1] / 2

        # 计算对角线长度
        diagonal_length = np.sqrt((bottom_right_x - top_left_x)**2 + (bottom_right_y - top_left_y)**2)

        # 对角线方向向量
        dx = (bottom_right_x - top_left_x) / diagonal_length
        dy = (bottom_right_y - top_left_y) / diagonal_length

        # 为7个roomba均匀分配位置
        num_roomba = 7
        for i in range(num_roomba):
            try:
                # 计算对角线上的位置比例（从0到1）
                ratio = (i + 0.5) / num_roomba
                
                # 计算相对于环境中心的位置
                roomba_rel_x = top_left_x + dx * diagonal_length * ratio
                roomba_rel_y = top_left_y + dy * diagonal_length * ratio

                # 计算绝对位置（加上环境中心）
                roomba_abs_x = roomba_rel_x + current_map_center[0]
                roomba_abs_y = roomba_rel_y + current_map_center[1]
                
                # 创建并发布roomba位置
                roomba_pose = Pose2D()
                roomba_pose.x = roomba_abs_x
                roomba_pose.y = roomba_abs_y
                roomba_pose.theta = 0.0  # 设置初始方向为0
                
                # 使用预创建的Publisher发布位置，避免重复创建Publisher
                if i < len(self.roomba_publishers) and self.roomba_publishers[i] is not None:
                    try:
                        self.roomba_publishers[i].publish(roomba_pose)
                        # 在每个roomba位置设置后稍作停顿，避免过度负载仿真器
                        rospy.sleep(0.5)
                    except Exception as e:
                        rospy.logwarn(f"无法发布到 /robot_{i+1}/cmd_pose: {e}")
                else:
                    rospy.logwarn(f"Publisher for /robot_{i+1}/cmd_pose not available")
                    
            except Exception as e:
                rospy.logwarn(f"设置roomba {i+1}位置时出错: {e}")
                continue  # 继续设置下一个roomba
        
        # 在所有位置设置完成后等待仿真器稳定
        rospy.sleep(1.0)

    def targetPointCheck(self, x, y):
        target_x = x
        target_y = y
        pass_flag = True
        x_pixel = int(target_x * self.R2P[0] + self.map_origin[0])
        y_pixel = int(target_y * self.R2P[1] + self.map_origin[1])
        window_size = int(self.target_size * np.amax(self.R2P))
        print(f"targetPointCheck: x_pixel={x_pixel}, y_pixel={y_pixel}, window_size={window_size}")
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
        #        if abs(target_x) < 2. and abs(target_y) < 2.:
        #            pass_flag = False
        print(f"targetPointCheck: pass_flag={pass_flag}")
        return pass_flag

    def robotPointCheck(self, x, y):
        target_x = x
        target_y = y
        pass_flag = True
        x_pixel = int(target_x * self.R2P[0] + self.map_origin[0])
        y_pixel = int(target_y * self.R2P[1] + self.map_origin[1])
        window_size_x1 = int(self.robot_range_x1 * np.amax(self.R2P))
        window_size_x2 = int(self.robot_range_x2 * np.amax(self.R2P))
        window_size_y = int(self.robot_range_y * np.amax(self.R2P))
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
        #        if abs(target_x) < 2. and abs(target_y) < 2.:
        #            pass_flag = False
        return pass_flag

    def LaserScanCallBack(self, scan):
        self.scan_param = [
            scan.angle_min,
            scan.angle_max,
            scan.angle_increment,
            scan.time_increment,
            scan.scan_time,
            scan.range_min,
            scan.range_max,
        ]
        self.scan = np.array(scan.ranges)
        self.laser_cb_num += 1
        # 记录激光数据更新时间
        self.last_scan_time = time.time()

    def OdometryCallBack(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion(
            [Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w]
        )
        self.state = [
            odometry.pose.pose.position.x,
            odometry.pose.pose.position.y,
            Euler[2],
        ]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def SimClockCallBack(self, clock):
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.0

    def GetImageObservation(self):
        # ros image to cv2 image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.image, "bgr8")
        except Exception as e:
            raise e
        # resize
        dim = (self.image_size[0], self.image_size[1])
        cv_resized_img = cv2.resize(cv_img, dim, interpolation=cv2.INTER_AREA)
        # cv2 image to ros image and publish
        try:
            resized_img = self.bridge.cv2_to_imgmsg(cv_resized_img, "bgr8")
        except Exception as e:
            raise e
        self.resized_ob.publish(resized_img)
        return cv_resized_img

    def GetLaserObservation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 10.0
        scan[np.isinf(scan)] = 10.0
        raw_beam_num = len(scan)
        #        print(raw_beam_num)
        sparse_beam_num = self.beam_num
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_right = []
        index = 0.0
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index += step
        sparse_scan_left = []
        index = raw_beam_num - 1.0
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate(
            (sparse_scan_right, sparse_scan_left[::-1]), axis=0
        )
        return scan_sparse

    def GetNoisyLaserObservation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 2.0
        nuniform_noise = np.random.uniform(-0.01, 0.01, scan.shape)
        linear_noise = np.multiply(np.random.normal(0.0, 0.01, scan.shape), scan)
        noise = nuniform_noise + linear_noise
        #        noise[noise > 0.03] = 0.03
        #        noise[noise < -0.03] = -0.03
        scan += noise
        scan[scan < 0.0] = 0.0
        # sample = random.sample(range(0, LAZER_BEAM), LAZER_BEAM/10)
        # scan[sample] = np.random.uniform(0.0, 1.0, LAZER_BEAM/10) * 30.
        return scan

    def GetSelfState(self):
        return self.state

    def GetSelfStateGT(self):
        return self.state_GT

    def GetSelfSpeedGT(self):
        return self.speed_GT

    def GetSelfSpeed(self):
        return self.speed

    def GetSimTime(self):
        return self.sim_time

    def ResetWorld(self, env_no, length1, length2, width):
        # 增加初始等待时间，确保仿真器稳定
        rospy.sleep(5.0)  # 从4.0增加到5.0
        
        try:
            self.past_actions = deque(maxlen=2)
            for initial_zero in range(2):
                self.past_actions.append(0)
            self.max_action[0] = 2.0
            self.max_action[1] = np.pi
            self.max_acc[0] = 2.5
            self.max_acc[1] = 3.2
            print("action bound is", self.max_action)
            self.length1 = length1
            self.length2 = length2
            self.width = width
            self.robot_range_x1 = self.length1 + 0.15
            self.robot_range_x2 = self.length2 + 0.15
            self.robot_range_y = self.width + 0.15
            self.stop_counter = 0.0
            self.crash_stop = False
            self.env = env_no
            self.map_size = self.map_sizes[env_no]
            if env_no < 2:
                self.target_size = 0.6
            elif env_no < 5:
                self.target_size = 0.4
            else:
                self.target_size = 0.3
            self.R2P = self.map_pixel / self.map_size
            
            # 添加异常处理的机器人位置设置
            try:
                self.set_robot_pose()
                rospy.sleep(1.0)  # 增加额外的等待时间
            except Exception as e:
                rospy.logwarn(f"设置机器人位置时出错: {e}")
                # 如果设置失败，等待更长时间后重试
                rospy.sleep(1.0)
                try:
                    self.set_robot_pose()
                except Exception as retry_e:
                    rospy.logerr(f"重试设置机器人位置仍失败: {retry_e}")
            
            self.stalled = False
            self.self_speed = [0.0, 0.0]
            self.step_target = [0.0, 0.0]
            self.step_r_cnt = 0.0
            self.ratio = 1.0
            self.start_time = time.time()
            
            # 增加最终等待时间
            rospy.sleep(3.0)  # 从3.0增加到4.0
            return self.max_action[0]
            
        except Exception as e:
            rospy.logerr(f"ResetWorld过程中发生错误: {e}")
            # 紧急情况下只重置基本参数
            self.crash_stop = False
            self.stalled = False
            self.self_speed = [0.0, 0.0]
            rospy.sleep(3.0)  # 给仿真器更多恢复时间
            return self.max_action[0]

    def Reset(self, env_no):
        rospy.sleep(3.0)
        self.past_actions = deque(maxlen=2)
        for initial_zero in range(2):
            self.past_actions.append(0)
        self.stop_counter = 0.0
        self.crash_stop = False
        self.env = env_no
        self.map_size = self.map_sizes[env_no]
        if env_no < 2:
            self.target_size = 0.6
        else:
            self.target_size = 0.4
        self.R2P = self.map_pixel / self.map_size
        #        self.set_robot_pose()
        self.stalled = False
        self.self_speed = [0.0, 0.0]
        self.step_target = [0.0, 0.0]
        self.step_r_cnt = 0.0
        self.ratio = 1.0
        self.start_time = time.time()
        rospy.sleep(3.0)
        return self.max_action[0]

    def Control(self, action):
        [v, w] = self.GetSelfSpeed()
        self.self_speed[0] = np.clip(
            action[0] * self.max_action[0],
            v - self.max_acc[0] * self.control_period,
            v + self.max_acc[0] * self.control_period,
        )
        #        print(self.self_speed[0])
        self.self_speed[1] = np.clip(
            action[1] * self.max_action[1],
            w - self.max_acc[1] * self.control_period,
            w + self.max_acc[1] * self.control_period,
        )
        #print(v)
        move_cmd = Twist()
        move_cmd.linear.x = self.self_speed[0]
        move_cmd.linear.y = 0.0
        move_cmd.linear.z = 0.0
        move_cmd.angular.x = 0.0
        move_cmd.angular.y = 0.0
        move_cmd.angular.z = self.self_speed[1]
        self.cmd_vel.publish(move_cmd)

    def set_robot_pose_test(self, k, env_index, shape_no):
        """设置机器人位置（测试用）"""
        # 获取目标环境的大小和中心
        env_size = self.map_sizes[env_index]
        env_center = self.map_center[env_index]
        
        x = self.test_initials[k, 0]
        y = self.test_initials[k, 1]
        
        robot_pose_data = Pose2D()
        robot_pose_data.x = x + env_center[0]
        robot_pose_data.y = y + env_center[1]
        robot_pose_data.theta = 0
        # robot_pose_data.theta = random.uniform(-np.pi, np.pi)  # 使用随机方向，与训练阶段保持一致
        self.pose_publisher.publish(robot_pose_data)
        # print(f"测试: 设置机器人位置 x={robot_pose_data.x:.3f}, y={robot_pose_data.y:.3f}, theta={robot_pose_data.theta:.3f}")
        # 为7个roomba设置位置，沿着当前环境的对角线均匀分布
        # 获取当前环境的大小和中心
        current_map_size = self.map_sizes[env_index]
        current_map_center = self.map_center[env_index]

        # 计算环境的左上角和右下角坐标（相对于环境中心）
        top_left_x = -current_map_size[0] / 2
        top_left_y = current_map_size[1] / 2
        bottom_right_x = current_map_size[0] / 2
        bottom_right_y = -current_map_size[1] / 2

        # 计算对角线长度
        diagonal_length = np.sqrt((bottom_right_x - top_left_x)**2 + (bottom_right_y - top_left_y)**2)

        # 对角线方向向量
        dx = (bottom_right_x - top_left_x) / diagonal_length
        dy = (bottom_right_y - top_left_y) / diagonal_length

        # 为7个roomba均匀分配位置
        num_roomba = 7
        for i in range(num_roomba):
            # 计算对角线上的位置比例（从0到1）
            ratio = (i + 0.5) / num_roomba
            
            # 计算相对于环境中心的位置
            roomba_rel_x = top_left_x + dx * diagonal_length * ratio
            roomba_rel_y = top_left_y + dy * diagonal_length * ratio
            
            # 计算绝对位置（加上环境中心）
            roomba_abs_x = roomba_rel_x + current_map_center[0]
            roomba_abs_y = roomba_rel_y + current_map_center[1]
            
            # 创建并发布roomba位置
            roomba_pose = Pose2D()
            roomba_pose.x = roomba_abs_x
            roomba_pose.y = roomba_abs_y
            roomba_pose.theta = 0.0  # 设置初始方向为0
            
            # 使用预创建的Publisher发布位置，避免重复创建Publisher
            if i < len(self.roomba_publishers) and self.roomba_publishers[i] is not None:
                try:
                    self.roomba_publishers[i].publish(roomba_pose)
                except Exception as e:
                    rospy.logwarn(f"无法发布到 /robot_{i+1}/cmd_pose: {e}")
            else:
                rospy.logwarn(f"Publisher for /robot_{i+1}/cmd_pose not available")
        return self.robot_size
    
    def GenerateTargetPoint_test(self, k, env_index, shape_no):
        """设置目标点（测试用）"""
        # 获取目标环境的中心
        env_center = self.map_center[env_index]
        
        # 使用测试目标点集合中的目标
        self.target_point = self.test_targets[k, :]
        x = self.target_point[0]
        y = self.target_point[1]
        
        # 调整到对应环境的坐标系
        self.target_point[0] = x + env_center[0]
        self.target_point[1] = y + env_center[1]
        
        # 计算距离
        self.pre_distance = np.sqrt(x**2 + y**2)
        self.distance = copy.deepcopy(self.pre_distance)

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop Moving")
        self.cmd_vel.publish(Twist())
        
        # 清理roomba Publishers
        rospy.loginfo("Cleaning up roomba publishers")
        for i, publisher in enumerate(self.roomba_publishers):
            if publisher is not None:
                try:
                    publisher.unregister()
                    rospy.loginfo(f"Unregistered publisher for /robot_{i+1}/cmd_pose")
                except Exception as e:
                    rospy.logwarn(f"Error unregistering publisher for /robot_{i+1}/cmd_pose: {e}")
        
        rospy.sleep(1)

    def goal_to_robot(self, goal_pose, robot_pose):
        # calculate the relative position of goal point in robotic frame

        #        print(self.pre_distance)
        theta = robot_pose[2]
        abs_x = goal_pose[0] - robot_pose[0]
        abs_y = goal_pose[1] - robot_pose[1]
        trans_matrix = np.matrix(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )
        rela = np.matmul(trans_matrix, np.array([[abs_x], [abs_y]]))
        rela_x = rela[0, 0]
        rela_y = rela[1, 0]
        rela_distance = np.maximum(np.sqrt(rela_x**2 + rela_y**2), 1e-8)
        rela_angle = np.arctan2(rela_y, rela_x)
        target_pose = [rela_y / rela_distance, rela_x / rela_distance, rela_distance]

        return target_pose

    def step(
        self,
    ):
        terminate = False
        # self.stop_counter = 0
        reset = 0
        goal_reach = 0  # 初始化goal_reach变量为0
        state = self.GetNoisyLaserObservation()
        laser_min = np.amin(state)
        state = np.reshape(state, (540))
        pool_state = np.zeros((90, 6))
        #        rot_m=np.matrix([[0,1], [-1, 0]])

        #        expand_p2 = np.array(np.matmul(expand_p1,rot_m))
        #        expand_p3 = np.array(np.matmul(expand_p2,rot_m))
        #        expand_p4 = np.array(np.matmul(expand_p3,rot_m))
        # 1080 obs_p scans, 360 degree
        for i in range(90):
            pool_state[i, 0] = np.cos(i * np.pi / 45.0 - np.pi / 2)
            pool_state[i, 1] = np.sin(i * np.pi / 45.0 - np.pi / 2)
            # dis needs to change for the pool_state[1]
            dis = np.min(state[6 * i : (6 * i + 6)])
            x_dis = pool_state[i, 0] * dis
            y_dis = pool_state[i, 1] * dis
            pool_state[i, 2] = dis
            pool_state[i, 3] = self.length1
            pool_state[i, 4] = self.length2
            pool_state[i, 5] = self.width
            if (
                abs(x_dis) <= self.width
                and y_dis <= self.length1
                and y_dis >= -self.length2
            ):
                self.stop_counter += 1.0

        #            if abs( pool_state[i,0])<=0.2 and abs(pool_state[i,1])<=0.2:
        #                self.stop_counter =1
        pool_state = np.reshape(pool_state, (540))
        [x, y, theta] = self.GetSelfStateGT()
        [v, w] = self.GetSelfSpeedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        #        print(self.pre_distance)
        abs_x = (self.target_point[0] - x) * self.ratio
        abs_y = (self.target_point[1] - y) * self.ratio
        trans_matrix = np.matrix(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )
        rela = np.matmul(trans_matrix, np.array([[abs_x], [abs_y]]))
        rela_x = rela[0, 0]
        rela_y = rela[1, 0]
        rela_distance = np.sqrt(rela_x**2 + rela_y**2)
        [x, y, theta] = self.GetSelfStateGT()
        [v, w] = self.GetSelfSpeed()
        self.distance = rela_distance
        rela_angle = np.arctan2(rela_y, rela_x)
        target_pose = [rela_distance, rela_angle]
        cur_act = [v * self.ratio, w]
        target = target_pose
        #        compare_state = self.robot_range/10.0*np.ones(540)
        state = np.concatenate(
            [
                pool_state,
                target_pose,
                cur_act,
                [
                    self.max_action[0],
                    self.max_action[1],
                    self.max_acc[0],
                    self.max_acc[1],
                ],
            ],
            axis=0,
        )
        #        print(state)

        reward = 2 * (self.pre_distance - self.distance)

        result = 0
        #        print(self.stop_counter)
        if self.stalled:
            self.stop_counter += 1.0
        #        else:
        #            self.stop_counter = 0

        if self.stop_counter >= 1.0:
            reward = -10.0
            terminate = True
            reset = 0
            goal_reach = 0  # 碰撞时goal_reach为0
            self.crash_stop = True
            #                print 'Crash end'
            self.stop_counter = 0
            self.stalled = False
            print("crashed")
        else:
            if self.distance < 0.2 and not self.stalled:
                reward = 10.0
                terminate = True
                reset = 1
                goal_reach = 1  # 到达目标时goal_reach为1
                #                print 'Reach the Goal'
                self.stop_counter = 0
                self.stalled = False

        #                self.goal_cancel_pub.publish(GoalID())

        return state, reward, terminate, goal_reach, self.distance, [x, y, theta]

    def GenerateTargetPoint(self, suc_rate):
        local_window = np.maximum(suc_rate * self.map_size[0], 0.5)
        # ensure all the target point spawned in the training map, i.e., the top left
        [xx, yy, theta] = self.GetSelfStateGT()
        xx = xx - self.map_center[self.env, 0]
        yy = yy - self.map_center[self.env, 1]
        
        # 计算有效范围
        x_min = max(-(self.map_size[0] / 2 - self.target_size), xx - local_window)
        x_max = min((self.map_size[0] / 2 - self.target_size), xx + local_window)
        y_min = max(-(self.map_size[1] / 2 - self.target_size), yy - local_window)
        y_max = min((self.map_size[1] / 2 - self.target_size), yy + local_window)
        
        # 检查是否有有效范围
        if x_min >= x_max or y_min >= y_max:
            print(f"WARNING: 目标点生成范围无效! x_range=[{x_min:.3f}, {x_max:.3f}], y_range=[{y_min:.3f}, {y_max:.3f}]")
            print(f"  机器人位置: [{xx:.3f}, {yy:.3f}], local_window: {local_window:.3f}")
            print(f"  地图尺寸: {self.map_size}, target_size: {self.target_size}")
            # 使用更大的范围作为备用
            x_min = -(self.map_size[0] / 2 - self.target_size)
            x_max = (self.map_size[0] / 2 - self.target_size)
            y_min = -(self.map_size[1] / 2 - self.target_size)
            y_max = (self.map_size[1] / 2 - self.target_size)
            
        # 当环境达到最高级别(4)时，添加调试信息
        if self.env == 4 and suc_rate > 0.8:
            print(f"[DEBUG] 最高级别地图(env=4)目标点生成: suc_rate={suc_rate:.3f}, "
                  f"local_window={local_window:.3f}, 地图尺寸={self.map_size[0]:.3f}")
            print(f"  机器人相对位置: [{xx:.3f}, {yy:.3f}]")
            print(f"  搜索范围: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
        
        # 初始目标点生成
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        
        # 添加重试次数限制，防止无限循环
        max_attempts = 500  # 最大尝试次数
        attempt_count = 0
        
        while not self.targetPointCheck(x, y) and not rospy.is_shutdown() and attempt_count < max_attempts:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            attempt_count += 1
            
            # 每50次尝试打印一次调试信息
            if attempt_count % 50 == 0:
                print(f"目标点生成尝试 {attempt_count}/{max_attempts}, 当前位置: [{x:.3f}, {y:.3f}]")
        
        # 如果达到最大尝试次数仍未找到有效位置
        if attempt_count >= max_attempts:
            print(f"WARNING: 目标点生成达到最大尝试次数({max_attempts})，使用最后生成的位置")
            print(f"  最终位置: [{x:.3f}, {y:.3f}], 检查结果: {self.targetPointCheck(x, y)}")
            # 强制扩大搜索范围，使用整个地图
            fallback_attempts = 100
            for _ in range(fallback_attempts):
                x = random.uniform(-(self.map_size[0] / 2 - self.target_size), 
                                 (self.map_size[0] / 2 - self.target_size))
                y = random.uniform(-(self.map_size[1] / 2 - self.target_size), 
                                 (self.map_size[1] / 2 - self.target_size))
                if self.targetPointCheck(x, y):
                    print(f"  备用搜索成功，使用位置: [{x:.3f}, {y:.3f}]")
                    break
            else:
                print(f"  备用搜索也失败，强制使用位置: [{x:.3f}, {y:.3f}]")
        
        self.target_point = [
            x + self.map_center[self.env, 0],
            y + self.map_center[self.env, 1],
        ]
        self.pre_distance = np.sqrt(x**2 + y**2)
        self.distance = copy.deepcopy(self.pre_distance)
        
        # 当环境达到最高级别(4)时，显示生成的目标点信息
        if self.env == 4 and suc_rate > 0.8:
            print(f"  生成目标点(相对): [{x:.3f}, {y:.3f}], 距离: {self.pre_distance:.3f}")
            print(f"  生成目标点(绝对): [{self.target_point[0]:.3f}, {self.target_point[1]:.3f}]")
            print(f"  尝试次数: {attempt_count}")
            
        # 记录生成统计信息
        if not hasattr(self, 'target_generation_stats'):
            self.target_generation_stats = {'total_attempts': 0, 'max_attempts_used': 0}
        self.target_generation_stats['total_attempts'] += attempt_count
        self.target_generation_stats['max_attempts_used'] = max(
            self.target_generation_stats['max_attempts_used'], attempt_count)
        
        # 每100次调用打印一次统计信息
        if not hasattr(self, 'generation_call_count'):
            self.generation_call_count = 0
        self.generation_call_count += 1
        if self.generation_call_count % 100 == 0:
            avg_attempts = self.target_generation_stats['total_attempts'] / self.generation_call_count
            print(f"目标点生成统计(调用{self.generation_call_count}次): "
                  f"平均尝试{avg_attempts:.1f}次, 最大尝试{self.target_generation_stats['max_attempts_used']}次")

    def GetLocalTarget(self):
        [x, y, theta] = self.GetSelfStateGT()
        [target_x, target_y] = self.target_point
        local_x = (target_x - x) * np.cos(theta) + (target_y - y) * np.sin(theta)
        local_y = -(target_x - x) * np.sin(theta) + (target_y - y) * np.cos(theta)
        return [local_x, local_y]

    def TargetPointCheck(self):
        target_x = self.target_point[0]
        target_y = self.target_point[1]
        pass_flag = True
        x_pixel = int(target_x * self.R2P[0] + self.map_origin[0])
        y_pixel = int(target_y * self.R2P[1] + self.map_origin[1])
        window_size = int(self.robot_range * np.amax(self.R2P))
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
        #        if abs(target_x) < 2. and abs(target_y) < 2.:
        #            pass_flag = False
        return pass_flag

    def Global2Local(self, path, pose):
        x = pose[0]
        y = pose[1]
        theta = pose[2]
        local_path = copy.deepcopy(path)
        for t in range(0, len(path)):
            local_path[t][0] = (path[t][0] - x) * np.cos(theta) + (
                path[t][1] - y
            ) * np.sin(theta)
            local_path[t][1] = -(path[t][0] - x) * np.sin(theta) + (
                path[t][1] - y
            ) * np.cos(theta)
        return local_path

    def ResetMap(self, path):
        self.map = copy.deepcopy(self.raw_map)
        target_point = path[-1]
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
        # x range
        if not isinstance(size, np.ndarray):
            x_range = [
                np.amax([int((point[0] - size / 2) * R2P[0]) + map_origin[0], 0]),
                np.amin(
                    [
                        int((point[0] + size / 2) * R2P[0]) + map_origin[0],
                        map_pixel[0] - 1,
                    ]
                ),
            ]

            y_range = [
                np.amax([int((point[1] - size / 2) * R2P[1]) + map_origin[1], 0]),
                np.amin(
                    [
                        int((point[1] + size / 2) * R2P[1]) + map_origin[1],
                        map_pixel[1] - 1,
                    ]
                ),
            ]
        else:
            x_range = [
                np.amax([int((point[0] - size[0] / 2) * R2P[0]) + map_origin[0], 0]),
                np.amin(
                    [
                        int((point[0] + size[0] / 2) * R2P[0]) + map_origin[0],
                        map_pixel[0] - 1,
                    ]
                ),
            ]

            y_range = [
                np.amax([int((point[1] - size[1] / 2) * R2P[1]) + map_origin[1], 0]),
                np.amin(
                    [
                        int((point[1] + size[1] / 2) * R2P[1]) + map_origin[1],
                        map_pixel[1] - 1,
                    ]
                ),
            ]

        for x in range(x_range[0], x_range[1] + 1):
            for y in range(y_range[0], y_range[1] + 1):
                # if map_img[map_pixel[1] - y - 1, x] < value:
                map_img[map_pixel[1] - y - 1, x] = value
        return map_img

    def DrawLine(self, point1, point2, value, map_img, map_pixel, map_origin, R2P):
        if point1[0] <= point2[0]:
            init_point = point1
            end_point = point2
        else:
            init_point = point2
            end_point = point1

        # transfer to map point
        map_init_point = [
            init_point[0] * R2P[0] + map_origin[0],
            init_point[1] * R2P[1] + map_origin[1],
        ]
        map_end_point = [
            end_point[0] * R2P[0] + map_origin[0],
            end_point[1] * R2P[1] + map_origin[1],
        ]
        # y = kx + b
        if map_end_point[0] > map_init_point[0]:
            k = (map_end_point[1] - map_init_point[1]) / (
                map_end_point[0] - map_init_point[0]
            )
            b = map_init_point[1] - k * map_init_point[0]
            if abs(k) < 1.0:
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
        [x, y, theta] = self.GetSelfStateGT()
        self.ResetMap(path)
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
        action_bound = self.max_action
        X = self.GetSelfState()
        X_t = self.GetLocalTarget() * np.array([self.ratio, self.ratio])
        P = np.array([1, 10.0])
        Ut = X_t * P

        if Ut[0] < -action_bound[0]:
            Ut[0] = -action_bound[0]
        elif Ut[0] > action_bound[0]:
            Ut[0] = action_bound[0]

        if Ut[1] < -action_bound[1]:
            Ut[1] = -action_bound[1]
        elif Ut[1] > action_bound[1]:
            Ut[1] = action_bound[1]
        Ut[0] = Ut[0]
        Ut[1] = Ut[1]
        #        print(self.self_speed[0])

        return Ut

    def OAController(self, action_bound, last_action):
        scan = (self.GetLaserObservation() + 0.5) * 10.0 - 0.19
        beam_num = len(scan)
        mid_scan = scan[int(beam_num / 4) : int(beam_num / 4) * 3]
        threshold = 1.2
        action = [last_action[0], 0.0]
        if np.amin(mid_scan) < threshold:
            if np.argmin(mid_scan) >= beam_num / 4:
                action[1] = -action_bound[1] * (
                    threshold - np.amin(mid_scan) / threshold
                )
            else:
                action[1] = action_bound[1] * (
                    threshold - np.amin(mid_scan) / threshold
                )

        if action[1] > action_bound[1]:
            action[1] = action_bound[1]
        elif action[1] < -action_bound[1]:
            action[1] = -action_bound[1]

        return [action]

    def ResetWorld_test(self, shape_no, env_index, k):
        """测试时重置世界环境，根据shape_no确定机器人参数"""
        rospy.sleep(2.0)
        
        # 重置past_actions - 与训练时保持一致
        self.past_actions = deque(maxlen=2)
        for initial_zero in range(2):
            self.past_actions.append(0)
        
        # 重置关键的控制参数 - 与训练时保持一致
        self.max_action[0] = 2.0
        self.max_action[1] = np.pi
        self.max_acc[0] = 2.5
        self.max_acc[1] = 3.2
        print("测试时action bound is", self.max_action)
        
        # 设置机器人参数 - 与训练时完全一致的逻辑
        # 首先使用与训练时相同的初始采样范围
        length1 = np.random.uniform(0.075, 0.6)
        length2 = np.random.uniform(0.075, 0.6)
        width = np.random.uniform(0.075, (length2 + length1) / 2.0)
        
        # 然后使用与训练时完全相同的约束逻辑
        if shape_no == 0:
            # 小尺寸机器人 (< 0.8) - 与训练时length_index==0的逻辑一致
            while length1 + length2 + width * 2 >= 0.8:
                length1 = np.random.uniform(0.075, 0.6)
                length2 = np.random.uniform(0.075, 0.6)
                width = np.random.uniform(0.075, (length2 + length1) / 2.0)
        else:
            # 其他尺寸组 - 与训练时length_index>0的逻辑一致
            while (length1 + length2 + width * 2 < (shape_no + 1.0) * 0.4 or 
                  length1 + length2 + width * 2 >= (shape_no + 2.0) * 0.4):
                length1 = np.random.uniform(0.075, 0.6)
                length2 = np.random.uniform(0.075, 0.6)
                width = np.random.uniform(0.075, (length2 + length1) / 2.0)
        
        # 添加调试输出来验证尺寸分布
        total_size = length1 + length2 + width * 2
        print(f"测试 shape_no={shape_no}, k={k}: length1={length1:.3f}, length2={length2:.3f}, width={width:.3f}, total_size={total_size:.3f}")
        
        # 设置机器人参数
        self.length1 = length1
        self.length2 = length2
        self.width = width
        self.robot_range_x1 = self.length1 + 0.15
        self.robot_range_x2 = self.length2 + 0.15
        self.robot_range_y = self.width + 0.15
        
        # 重置其他重要状态 - 与训练时保持一致
        self.stop_counter = 0.0
        self.crash_stop = False
        self.env = env_index
        self.map_size = self.map_sizes[env_index]
        
        # 根据环境设置目标大小
        if env_index < 2:
            self.target_size = 0.6
        elif env_index < 5:
            self.target_size = 0.4
        else:
            self.target_size = 0.3
            
        self.R2P = self.map_pixel / self.map_size
        
        # 设置机器人位置 - 使用测试专用的set_robot_pose_test方法
        self.set_robot_pose_test(k, env_index, shape_no)
        
        # 重置机器人状态 - 正确的变量名是stalled，不是stalls
        self.stalled = False
        self.self_speed = [0.0, 0.0]
        self.step_target = [0.0, 0.0]
        self.step_r_cnt = 0.0
        self.ratio = 1.0
        self.start_time = time.time()
        
        # 等待环境稳定
        rospy.sleep(2.0)
        
        # 返回最大速度 - 与训练时保持一致
        return self.max_action[0]

    def check_stage_health(self):
        """检查Stage仿真器健康状态"""
        try:
            # 检查激光数据是否更新
            if hasattr(self, 'last_scan_time'):
                current_time = time.time()
                if current_time - self.last_scan_time > 5.0:  # 5秒无数据更新
                    print("WARNING: 激光数据超过5秒未更新")
                    return False
            
            # 检查机器人位置是否有变化
            if hasattr(self, 'last_position_check'):
                current_pos = self.GetSelfStateGT()
                if hasattr(self, 'last_position') and current_pos:
                    pos_diff = ((current_pos[0] - self.last_position[0])**2 + 
                               (current_pos[1] - self.last_position[1])**2)**0.5
                    if pos_diff < 0.001 and time.time() - self.last_position_check > 10.0:
                        print("WARNING: 机器人位置超过10秒未变化")
                        return False
                self.last_position = current_pos
                self.last_position_check = time.time()
            else:
                self.last_position_check = time.time()
                self.last_position = self.GetSelfStateGT()
            
            # 检查ROS节点是否还在运行
            if rospy.is_shutdown():
                print("WARNING: ROS节点已关闭")
                return False
            
            # 检查scan数据是否有效
            if self.scan is not None:
                if len(self.scan) == 0 or np.all(np.isnan(self.scan)):
                    print("WARNING: 激光扫描数据无效")
                    return False
            else:
                print("WARNING: 激光扫描数据为None")
                return False
                
            return True
            
        except Exception as e:
            rospy.logwarn(f"健康检查出错: {e}")
            return False


#    def GoalPublish(self, pose):
#        x = pose[0]
#        y = pose[1]
#        yaw = pose[2]
#
#        self.move_base_goal.header.frame_id = "map"
#        self.move_base_goal.header.stamp = rospy.Time()
#        self.move_base_goal.pose.position.x = x
#        self.move_base_goal.pose.position.y = y
#        self.move_base_goal.pose.position.z = 0.
#        quaternion = tf.transformations.quaternion_from_euler(0., 0., yaw)
#        self.move_base_goal.pose.orientation.x = quaternion[0]
#        self.move_base_goal.pose.orientation.y = quaternion[1]
#        self.move_base_goal.pose.orientation.z = quaternion[2]
#        self.move_base_goal.pose.orientation.w = quaternion[3]
#        self.goal_pub.publish(self.move_base_goal)
