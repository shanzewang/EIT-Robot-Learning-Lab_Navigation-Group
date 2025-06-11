#!/bin/bash
# 保存为run_spn.sh

# 设置ROS环境和stageros1
source /opt/ros/noetic/setup.bash
source ~/sz_dclp_ws/devel/setup.bash   #! 需要改到自己的stage 文件夹下

# 正确初始化conda
eval "$(conda shell.bash hook)"
conda activate dclpv2   # ! 改为自己的虚拟环境名称

# 定义清理函数
cleanup() {
    echo "正在清理相关进程..."
    
    # 杀死Python程序
    if [ ! -z "$PYTHON_PID" ]; then
        echo "关闭 Python 程序 (PID: $PYTHON_PID)"
        kill -TERM $PYTHON_PID 2>/dev/null
        sleep 1
        kill -9 $PYTHON_PID 2>/dev/null
    fi
    
    # 通过Python程序启动的进程，需要用不同方法查找和清理
    echo "查找并清理相关ROS进程..."
    
    # 查找并杀死stageros（通过world文件名识别）
    STAGEROS_PIDS=$(pgrep -f "stageros.*d8888153.world")
    if [ ! -z "$STAGEROS_PIDS" ]; then
        echo "关闭 stageros 进程: $STAGEROS_PIDS"
        echo $STAGEROS_PIDS | xargs kill -TERM 2>/dev/null
        sleep 1
        echo $STAGEROS_PIDS | xargs kill -9 2>/dev/null
    fi
    
    # 查找并杀死roscore（通过端口范围识别）
    ROSCORE_PIDS=$(pgrep -f "roscore.*-p.*1[0-5][0-9][0-9][0-9]")
    if [ ! -z "$ROSCORE_PIDS" ]; then
        echo "关闭 roscore 进程: $ROSCORE_PIDS"
        echo $ROSCORE_PIDS | xargs kill -TERM 2>/dev/null
        sleep 1
        echo $ROSCORE_PIDS | xargs kill -9 2>/dev/null
    fi
    
    echo "✅ 清理完成"
}

# 启动Python程序并记录PID
echo "🐍 启动 Python 程序..."
python torchdclp_train_true.py &
PYTHON_PID=$!

# 捕获Ctrl+C并调用清理函数
trap "cleanup; exit" SIGINT SIGTERM

# 等待Python程序结束
wait $PYTHON_PID

# 程序结束后清理
cleanup
exit 0