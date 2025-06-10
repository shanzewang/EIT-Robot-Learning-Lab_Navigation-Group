#!/bin/bash
# 保存为run_spn.sh

# 设置ROS环境和stageros1
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

# 正确初始化conda
eval "$(conda shell.bash hook)"
conda activate dclpv2

# 定义清理函数
cleanup() {
    echo "正在清理所有相关进程..."
    pkill -9 -f sac_torch.py
    pkill -9 stage
    pkill -9 -f stageros
    pkill -9 -f roscore
    echo "完全终止程序"
}

# 启动Python程序
python sac_torch.py &
PID=$!

# 捕获Ctrl+C并调用清理函数
trap "cleanup; exit" SIGINT SIGTERM

# 等待Python程序结束
wait $PID

# 无论程序是正常结束还是出错结束，都会执行清理操作
cleanup
exit 0

