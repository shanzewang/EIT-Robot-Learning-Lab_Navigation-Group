# MAER-Nav 项目汇总
- core_denseu_ada4_goal.py为训练网络文件
- IPAPTrain network pi2 bcar 0.2freq10 0221*map20*20 32 文件夹存放训练好的文件，其中效果最好的为36编号
- sacb lPAPtrain 0.7suc pi2bc20*20 32d6.py为训练执行文件
- stage back nodis 0212.py为环境文件
- sacblPAP0221pi2bc20*20 32 d6.world生成stage地图以及机器人，其中机器人配置与大车相同
- d6.jpg为训练环境图片，加入了少量的local minimum来应对类似情况
- IPAPRec Train为IPAPRec训练原文件
