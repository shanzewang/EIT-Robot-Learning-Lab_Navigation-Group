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

        self.scan = None
        self.beam_num = beam_num
        self.laser_cb_num = 0
        #        self.robot_range_bound = 0.1
        self.rot_counter = 0
        # sdjisdj
        #! sdjiajdaj
        # ?sdjiasdjsoa
        # * sdadsa
        # TODO djisaodj

        self.now_phase = 1
        self.next_phase = 4
        self.step_target = [0.0, 0.0]
        self.step_r_cnt = 0.0
        self.stop_counter = 0
        #        self.action_table = [[-1.,-1.],[0.0,-1.],[1.,-1.],[-1.,0.],[0.,0.],[1.,0.],[1.,1.],[0.,1.],[-1.,1.]]
        self.max_action = [0.5, np.pi / 2]
        self.min_action = [0.0, -np.pi / 2]
        self.ratio = 1.0

        self.self_speed = [0.3 / self.ratio, 0.0]
        self.target_point = [0, 5.5]
        map_img = cv2.imread('d6.jpg', 0)
        ret, binary_map = cv2.threshold(map_img, 10, 1, cv2.THRESH_BINARY)
        binary_map = 1 - binary_map
        self.map = binary_map.astype(np.float32)
        # cv2.imshow('img',binary_map*255)
        # cv2.waitKey(0)
        height, width = binary_map.shape
        self.map_pixel = np.array([width, height])
        self.map_sizes = np.zeros((25, 2))

        # ! change 0211
        # for map_no in range(25):
        #     self.map_sizes[map_no, 0] = 16.0 * (0.8 ** int(map_no / 5))
        #     self.map_sizes[map_no, 1] = 16.0 * (0.8 ** int(map_no % 5))
        # 地图尺寸计算
        for map_no in range(25):
            self.map_sizes[map_no, 0] = 20.0 * (0.8 ** int(map_no / 5))
            self.map_sizes[map_no, 1] = 20.0 * (0.8 ** int(map_no % 5))

        # for map_no in range(25):
        #     self.map_sizes[map_no, 0] = 8.0 * (0.8 ** int(map_no / 5))
        #     self.map_sizes[map_no, 1] = 8.0 * (0.8 ** int(map_no % 5))
        self.map_origin = self.map_pixel / 2 - 1
        # self.robot_size = 0.38
        self.robotsize = 0.35
        self.target_size = 0.4
        # self.robot_range_x = 0.35
        # self.robot_range = 0.2
        # ! change 0211
        self.robot_range_x = 0.62
        self.robot_range = 0.2
        self.robot_range_y = 0.64
        self.map_center = np.zeros((25, 2))

        # # ! change 0211\
        # for map_no in range(25):
        #     self.map_center[map_no, 0] = -32 + int(map_no / 5) * 16.0
        #     self.map_center[map_no, 1] = 32 - int(map_no % 5) * 16.0

        # 地图中心点计算
        for map_no in range(25):
            self.map_center[map_no, 0] = -40 + int(map_no / 5) * 20.0
            self.map_center[map_no, 1] = 40 - int(map_no % 5) * 20.0
        # for map_no in range(25):
        #     self.map_center[map_no, 0] = -16 + int(map_no / 5) * 8.0
        #     self.map_center[map_no, 1] = 16 - int(map_no % 5) * 8.0
        self.robot_value = 0.33
        self.target_value = 0.66
        self.path_value = 0.1
        self.env = 0

        # -----------Publisher and Subscriber-------------
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=100)
        self.pose_publisher = rospy.Publisher('cmd_pose', Pose2D, queue_size=1000)
        rospy.loginfo("Publisher Created: /cmd_pose")

        self.object_state_sub = rospy.Subscriber(
            'base_pose_ground_truth', Odometry, self.GroundTruthCallBack
        )
        self.laser_sub = rospy.Subscriber(
            'base_scan', LaserScan, self.LaserScanCallBack
        )
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.OdometryCallBack)
        self.sim_clock = rospy.Subscriber('clock', Clock, self.SimClockCallBack)
        #        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        #        self.goal_cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=10)

        # -----------Service-------------------
        self.ResetStage = rospy.ServiceProxy('reset_positions', std_srvs.srv.Empty)
        self.stalls = rospy.Subscriber("/stalled", Int8, self.update_robot_stall_data)

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
        self.pose_publisher.publish(robot_pose_data)

    def targetPointCheck(self, x, y):
        target_x = x
        target_y = y
        pass_flag = True
        x_pixel = int(target_x * self.R2P[0] + self.map_origin[0])
        y_pixel = int(target_y * self.R2P[1] + self.map_origin[1])
        window_size = int(self.target_size * np.amax(self.R2P))
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

    def robotPointCheck(self, x, y):
        target_x = x
        target_y = y
        pass_flag = True
        x_pixel = int(target_x * self.R2P[0] + self.map_origin[0])
        y_pixel = int(target_y * self.R2P[1] + self.map_origin[1])
        window_size_x = int(self.robot_range_x * np.amax(self.R2P))
        window_size_y = int(self.robot_range_y * np.amax(self.R2P))
        for x in range(
            np.amax([0, x_pixel - window_size_x]),
            np.amin([self.map_pixel[0] - 1, x_pixel + window_size_x]),
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
        scan[np.isnan(scan)] = 10.0
        #        nuniform_noise = np.random.uniform(-0.01, 0.01, scan.shape)
        #        linear_noise = np.multiply(np.random.normal(0., 0.03, scan.shape), scan)
        #        noise = nuniform_noise + linear_noise
        ##        noise[noise > 0.01] = 0.01
        ##        noise[noise < -0.01] = -0.01
        #        scan += noise
        #        scan[scan < 0.] = 0.
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

    def ResetWorld(self, env_no):
        rospy.sleep(3.0)
        self.past_actions = deque(maxlen=2)
        for initial_zero in range(2):
            self.past_actions.append(0)
        self.stop_counter = 0.0
        self.crash_stop = False
        self.env = env_no
        self.map_size = self.map_sizes[env_no]
        self.R2P = self.map_pixel / self.map_size
        self.set_robot_pose()
        self.stalls
        self.self_speed = [0.0, 0.0]
        self.step_target = [0.0, 0.0]
        self.step_r_cnt = 0.0
        self.ratio = 1.0
        self.start_time = time.time()
        rospy.sleep(2.0)
        return self.robot_range

    def Reset(self, env_no):
        rospy.sleep(3.0)
        self.past_actions = deque(maxlen=2)
        for initial_zero in range(2):
            self.past_actions.append(0)
        self.stop_counter = 0.0
        self.crash_stop = False
        self.env = env_no
        self.map_size = self.map_sizes[env_no]
        self.R2P = self.map_pixel / self.map_size
        #        self.set_robot_pose()
        self.stalls
        self.self_speed = [0.0, 0.0]
        self.step_target = [0.0, 0.0]
        self.step_r_cnt = 0.0
        self.ratio = 1.0
        self.start_time = time.time()
        rospy.sleep(2.0)
        return self.robot_range

    def Control(self, action):
        self.self_speed[0] = action[0] * self.max_action[0]
        #        print(self.self_speed[0])
        self.self_speed[1] = action[1] * self.max_action[1]
        move_cmd = Twist()
        move_cmd.linear.x = self.self_speed[0]
        move_cmd.linear.y = 0.0
        move_cmd.linear.z = 0.0
        move_cmd.angular.x = 0.0
        move_cmd.angular.y = 0.0
        move_cmd.angular.z = self.self_speed[1]
        self.cmd_vel.publish(move_cmd)

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop Moving")
        self.cmd_vel.publish(Twist())
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
        # target_pose = [rela_y / rela_distance, rela_x / rela_distance, rela_distance]
        target_pose = [rela_distance, rela_angle]

        return target_pose

    def step(
        self,
    ):
        terminate = False
        reset = 0
        state = self.GetNoisyLaserObservation()
        laser_min = np.amin(state)
        state = np.reshape(state, (1667))

        state = state[:1664]

        # 对原始的1664个点进行64组的池化
        pool_state = np.zeros(32)
        points_per_group = 52  # 1664/64 = 26，正好整除
        for i in range(32):
            pool_state[i] = np.min(
                state[i * points_per_group : (i + 1) * points_per_group]
            )

        if laser_min <= 0.2:
            self.stop_counter = 1

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
        self.distance = rela_distance
        rela_angle = np.arctan2(rela_y, rela_x)
        target_pose = [rela_distance, rela_angle]
        cur_act = [v * self.ratio, w]

        state = np.concatenate([pool_state, target_pose, cur_act], axis=0)
        # !                        32           2           2

        reward = 2 * (self.pre_distance - self.distance) - 0.01

        if self.stalled:
            self.stop_counter += 1.0
            reward = -1.0
            terminate = True
            print('Crash')

        # ! -----changed by shanze 25.2.5  最大允许碰撞次数1次，碰撞奖励设为-10 ----------
        if self.stop_counter >= 1.0:
            reward = -10.0
            terminate = True
            reset = 0
            self.crash_stop = True
            print('Crash end')
            self.stop_counter = 0
            self.stalled = False
        else:
            if self.distance < 0.2 and not self.stalled:
                reward = 10.0
                terminate = True
                reset = 1
                print('Reach the Goal')
                self.stop_counter = 0
                self.stalled = False

        return state, reward, terminate, reset, self.distance, [x, y, theta]

    def GenerateTargetPoint(self, suc_rate):
        suc_rate = np.maximum(suc_rate, 0.1)
        # ensure all the target point spawned in the training map, i.e., the top left
        [xx, yy, theta] = self.GetSelfStateGT()
        xx = xx - self.map_center[self.env, 0]
        yy = yy - self.map_center[self.env, 1]
        #        x = random.uniform(-(self.map_size[0]/2 - self.target_size), self.map_size[0]/2 - self.target_size)
        #        y = random.uniform(-(self.map_size[1]/2 - self.target_size), self.map_size[1]/2 - self.target_size)
        x = random.uniform(
            max(-(self.map_size[0] / 2 - self.target_size), xx - 5.0 * suc_rate),
            min((self.map_size[0] / 2 - self.target_size), xx + 5.0 * suc_rate),
        )
        y = random.uniform(
            max(-(self.map_size[1] / 2 - self.target_size), yy - 5.0 * suc_rate),
            min((self.map_size[1] / 2 - self.target_size), yy + 5.0 * suc_rate),
        )
        while not self.targetPointCheck(x, y) and not rospy.is_shutdown():
            x = random.uniform(
                max(-(self.map_size[0] / 2 - self.target_size), xx - 5.0 * suc_rate),
                min((self.map_size[0] / 2 - self.target_size), xx + 5.0 * suc_rate),
            )
            y = random.uniform(
                max(-(self.map_size[1] / 2 - self.target_size), yy - 5.0 * suc_rate),
                min((self.map_size[1] / 2 - self.target_size), yy + 5.0 * suc_rate),
            )
        self.target_point = [
            x + self.map_center[self.env, 0],
            y + self.map_center[self.env, 1],
        ]
        self.pre_distance = np.sqrt(x**2 + y**2)
        self.distance = copy.deepcopy(self.pre_distance)

    def GenerateTargetPoint_test(self, i):
        # independent testing point in different maps
        #        test_targets = [[2,2.5],[1.7,-2.35],[-1.75,-2.0],[-2,2.6]]
        #        test_targets = [[-2.4,4.00],[-4.9,5.4],[5.75,6],[3.0,7.53],[-2.44,-4],[-1.5,-1.5],[5.96,-2.64],[3.74,-1.34]]
        targets = 5.0 / 3.5 * np.array([[3.05, -3], [3.05, 3], [-2, -3], [-2.5, 3.0]])
        test_targets0 = targets + self.map_center[0, :]
        test_targets1 = 5.0 * 0.8 / 4.0 * targets + self.map_center[1, :]
        test_targets2 = 5.0 * (0.8**1) / 4.0 * targets + self.map_center[2, :]
        test_targets3 = 5.0 * (0.8**3) / 4.0 * targets + self.map_center[3, :]
        test_targets4 = 5.0 * (0.8**2) / 4.0 * targets + self.map_center[4, :]
        test_targets5 = 5.0 * (0.8**5) / 4.0 * targets + self.map_center[5, :]
        test_targets6 = 5.0 * (0.8**6) / 4.0 * targets + self.map_center[6, :]
        test_targets7 = 5.0 * (0.8**7) / 4.0 * targets + self.map_center[7, :]
        test_targets = np.concatenate(
            [
                test_targets0,
                test_targets1,
                test_targets2,
                test_targets3,
                test_targets4,
                test_targets5,
                test_targets6,
                test_targets7,
            ],
            axis=0,
        ).tolist()

        #        self.target_point = test_targets[np.random.choice([0,1,2,3])]
        self.target_point = test_targets[i]
        print(self.target_point)
        x = self.target_point[0]
        y = self.target_point[1]
        self.pre_distance = np.sqrt(x**2 + y**2)
        self.distance = copy.deepcopy(self.pre_distance)

    def set_robot_pose_test(self, i):
        test_initials = np.reshape(np.tile(self.map_center, 4), [32, 2]).tolist()
        robot_pose_data = Pose2D()
        x = test_initials[i][0]
        y = test_initials[i][1]
        #        while not self.robotPointCheck(x,y) and not rospy.is_shutdown():
        #            x = random.uniform(-(self.map_size[0]/2 - self.target_size), self.map_size[0]/2 - self.target_size)
        #            y = random.uniform(-(self.map_size[1]/2 - self.target_size), self.map_size[1]/2 - self.target_size)
        robot_pose_data.theta = 0
        robot_pose_data.x = x
        robot_pose_data.y = y
        self.pose_publisher.publish(robot_pose_data)
        rospy.sleep(2.0)

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
        Ut[0] = Ut[0] / self.max_action[0]
        Ut[1] = Ut[1] / self.max_action[1]
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
