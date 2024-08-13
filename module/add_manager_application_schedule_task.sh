#!/bin/bash

# owner: songquanheng
# date: 2020年11月23日10:06:08于文福路9号
# 警告：管理节点的/etc/network/interfaces中仅有一行address ip,该脚本才能正确工作。
# 脚本作用： 该脚本用于把定时运维脚本module/manager_schdule_task.sh
#        自动加入管理节点crontab服务中
# 8090 前端主程序
# 8080 后端主程序
# 5000 服务管理程序
# 15001 pbs_server端口
# 15002 pbs_mom框架程序端口
# 15003 注意：pbs_mom占据两个端口
# 15004 pbs_sched框架程序端口
# 9200  es数据检索与外部通讯端口
# 9300  数据检索节点内部通讯端口
# 9400  检索平台端口
# 5236  数据库端口
# 8888  node端口，实时预览
# 1937  视频拼接使用端口

cd $(dirname $0)
pwd
. ../util/util_function.sh
. ../conf/camp.conf

printGreenText "为管理节点${manager_ip}添加定时任务，维持管理节点上基本服务正常"

cron_job="*/1 * * * * . /etc/profile; /bin/bash $(pwd)/manager_schedule_task.sh"
crontab -l > temp.conf && echo "${cron_job}" >> temp.conf && crontab temp.conf && rm -rf temp.conf

printRedText '当前管理节点IP为：$(ip),拥有如下的定时任务'
crontab -l | grep -v '#'

if crontab -l | grep "manager_schedule_task" > /dev/null ;then
	printRedText '在定时任务中已经有了manager_schedule_task'
	crontab -l | grep 'manager_schedule_task'
else
	printGreenText "安装管理节点应用程序运维定时任务"
	crontab -l > app.cron
	echo "${cron_job}" >> app.cron
	crontab app.cron
	rm -rf app.cron
fi

printRedText "当前管理节点IP为：${manager_ip},拥有如下的定时任务"