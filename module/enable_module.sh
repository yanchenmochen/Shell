#!/bin/bash
# author：songquanheng
# 2020-04-07 10:04:39
# 脚本用途： 该脚本用于配置开机自启动程序



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

printGreenText "配置开机启动实时预览程序"
systemctl enable campus-preview.service


printGreenText "配置开机启动AI推理服务主程序"
sleep 2
systemctl enable campus

printGreenText "配置开机启动启动页面程序"
systemctl enable campus-front.service

# printGreenText "启动设备发现程序"
# systemctl enable campus-discovery
# systemctl start campus-discovery

printGreenText "配置开机启动启动运行管理中心程序"
systemctl enable campus-monitor


printGreenText "配置开机启动启动安防分析算法"
systemctl enable campus-algorithm


printGreenText "配置开机启动启动IOT"
systemctl enable campus-iot


printGreenText "中电五十二所AI推理启动完成"
sleep 2
