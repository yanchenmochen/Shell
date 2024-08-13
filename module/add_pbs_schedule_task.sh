#!/bin/bash

# owner: songquanheng
# date: 2020年11月17日16:24:19
# 警告： 需要已经处理过每个板子上多余的usb0网卡信息。这样在调用util中函数ip才能正常工作
# 警告： A200服务每次重启之后，
#		/var的权限会发生变化从0555变为777，
#		但PBS的运行需要其为0555
# 脚本作用： 该脚本用于为A200配置定时任务以检测pbs_mom和trquauthd服务。

cd $(dirname $0)
pwd
. ../util/util_function.sh
. ../conf/camp.conf

printGreenText "为芯片$(ip)添加定时任务，维持pbs_mom和trqquthd服务"

cron_job="*/1 * * * * . /etc/profile; /bin/sh $(pwd)/schedule_pbs.sh"
crontab -l > temp.conf && echo "${cron_job}" >> temp.conf && crontab temp.conf && rm -rf temp.conf

printRedText '当前A200IP为：$(ip),拥有如下的定时任务'
crontab -l | grep -v '#'

if crontab -l | grep "schedule_pbs" > /dev/null ;then
	printRedText '在定时任务中已经有了设备发现任务schedule_device_discovery'
	crontab -l | grep 'schedule_pbs'
else
	printGreenText "安装PBS状态维护定时任务"
	crontab -l > pbs.cron
	echo "${cron_job}" >> pbs.cron
	crontab pbs.cron
	rm -rf pbs.cron
fi

printRedText "当前A200IP为：$(ip),拥有如下的定时任务"
crontab -l
