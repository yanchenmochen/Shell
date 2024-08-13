#!/bin/bash
# author：songquanheng
# 2020-12-30 09:23:25
# 脚本用途： 该脚本用于卸载运行管理中心程序
# 1、从camp.conf获取后端程序安装路径
# 2、停止运行管理中心服务
# 3、移除设备发现配置文件campus-monitor.service文件

cd $(dirname $0)

. ../util/util_function.sh
printGreenText "导入功能函数，卸载运行管理程序程序"

# 导入配置文件，获取安装路径
. ../conf/camp.conf


if grep 'Kylin' /etc/lsb-release > /dev/null; then
	printGreenText "当前系统为Kylin操作系统"
else
	printGreenText "当前系统为A200加速卡，系统为Ubuntu"
fi
sleep 2
printGreenText "停止运行管理中心服务"

systemctl stop campus-monitor
sleep 1
printGreenText "移除设备发现相关配置"
rm -rf  /etc/systemd/system/campus-monitor.service
systemctl daemon-reload

printGreenText "运行管理中心服务移除成功"
