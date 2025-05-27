#!/bin/bash
# 测试脚本 run_test.sh

# 设置ROS环境
source /opt/ros/noetic/setup.bash

# 初始化conda环境
eval "$(conda shell.bash hook)"
conda activate dclpv2

# 定义清理函数
cleanup() {
    echo "正在清理所有相关进程..."
    pkill -9 -f test_dclp_lidar.py
    pkill -9 stage
    pkill -9 -f stageros
    pkill -9 -f roscore
    echo "完全终止程序"
}

# 启动Python测试程序
python test_dclp_lidar.py &
PID=$!

# 捕获Ctrl+C并调用清理函数
trap "cleanup; exit" SIGINT SIGTERM

# 等待Python程序结束
wait $PID

# 无论程序是正常结束还是出错结束，都会执行清理操作
cleanup
exit 0 