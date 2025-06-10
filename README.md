使用run_dynamic.sh开始训练

该版本完善了整个attention的动态网络，包括spatial，temporal和最后mlp整合

该版本在仿真训练之前发布，因此所有的动态数据均为硬编码，需要在环境和前端整合这些部分才能完善后续的使用，目前调用的是env传参

def get_action(o, env=None ,deterministic=False):动态信息在env中获取，目前为了代码运行和网络测试，我设置了None

可以先测试用
