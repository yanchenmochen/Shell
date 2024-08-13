#!/bin/bash

# owner: songquanheng
# date: 2020年11月17日16:57:39
# 警告：需要已经处理过每个板子上多余的usb0网卡信息。这样在调用util中函数ip才能正常工作
# 脚本作用： 该脚本用于为A200配置定时任务以维持设备发现程序始终处于存活状态。
# 			该脚本叶可以在管理节点上添加定时任务

if [ `whoami` != 'root' ]; then
	echo "please login in with root user"
	exit 5
fi
cd $(dirname $0)
pwd
. ../util/util_function.sh
. ../conf/camp.conf
printGreenText "添加定时设备发现任务，维持设备发现服务"

cron_job="*/1 * * * * . /etc/profile; /bin/sh $(pwd)/schedule_device_discovery.sh"

if crontab -l | grep "schedule_device_discovery" > /dev/null ;then
	printRedText '在定时任务中已经有了设备发现任务schedule_device_discovery'
	crontab -l | grep 'schedule_device_discovery'
else
	printGreenText "安装设备发现定时任务"
	crontab -l > device.cron
	echo "${cron_job}" >> device.cron
	crontab device.cron
	rm -rf device.cron
fi

