#!/bin/bash
# author：songquanheng
# 2021年7月15日20:40:09
# 脚本用途： 该脚本用于启动程序



cd $(dirname $0)
pwd
. ../util/util_function.sh
. ../conf/camp.conf


printGreenText "管理节点IP: ${manager_ip}"
printGreenText "启动有关程序"
systemctl daemon-reload
# printGreenText "启动elasticsearch服务"
# systemctl enable elasticsearch
# systemctl restart elasticsearch

# sleep 2
# printGreenText "启动search检索服务"
# systemctl enable searchx
# systemctl restart searchx


printGreenText "启动实时预览程序"
sleep 2
systemctl restart campus-preview.service

printGreenText "启动AI推理服务主程序"
sleep 2
systemctl restart campus


printGreenText "启动页面程序"
sleep 2
systemctl restart campus-front.service

# printGreenText "启动设备发现程序"
# systemctl enable campus-discovery
# systemctl start campus-discovery


printGreenText "启动运行管理中心程序"
sleep 2
systemctl restart campus-monitor


printGreenText "启动安防分析算法"
sleep 2
systemctl restart campus-algorithm


printGreenText "启动IOT"
sleep 2
systemctl restart campus-iot

printGreenText "中电五十二所AI推理启动完成"
sleep 2
