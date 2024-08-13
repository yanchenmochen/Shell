#!/bin/bash
# author：songquanheng
# 2021年1月12日16:08:19
# 脚本用途： 该脚本用于卸载安防分析算法
# 1、从camp.conf获取后端程序安装路径
# 2、停止安防分析算法
# 3、移除设备发现配置文件campus-algorithm.service文件

cd $(dirname $0)

. ../util/util_function.sh
printGreenText "导入功能函数，卸载设备发现程序"

# 导入配置文件，获取安装路径
. ../conf/camp.conf


if grep 'Kylin' /etc/lsb-release > /dev/null; then
	printGreenText "当前系统为Kylin操作系统"
else
	printGreenText "当前系统为A200加速卡，系统为Ubuntu"
fi
sleep 2
printGreenText "停止安防分析服务"


systemctl disable campus-algorithm
systemctl stop campus-algorithm
sleep 1
printGreenText "移除安防分析相关配置"
rm -rf  /etc/systemd/system/campus-algorithm.service
systemctl daemon-reload

printGreenText "设备发现服务移除成功"
