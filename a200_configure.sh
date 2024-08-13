#!/bin/bash

# owner: songquanheng
# date: 2020年11月18日16:16:18于文福路9号
# 警告：需要已经处理过每个板子上多余的usb0网卡信息。这样在调用util中函数ip才能正常工作
# 警告：sh a200_pre_configure.sh iuni7已经成功执行过，即每个板子的域名已经成功修改。
# 警告：
# 脚本作用： 
# 1. 安装torque程序并添加环境变量
# 2. 修改pbs配置文件
# 3. 添加用户HPCUser
# 4. 为板子添加dns信息
# 5. 为板子添加定时任务

cd $(dirname $0)
pwd
. ./util/util_function.sh
. ./conf/camp.conf

desc() {
    printRedText "脚本的主要作用包括："
	printGreenText "1. 安装torque程序并添加环境变量"
	printGreenText "2. 修改pbs配置文件"
	printGreenText "3. 添加用户HPCUser"
	printGreenText "4. 为板子添加dns信息"
	printGreenText "5. 为板子添加定时任务"
}

if [ `whoami` != 'root' ]; then
	echo "please login in with root user"
	exit 5
fi

if [ "$1" = "-help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "--help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "help" ] ; then
        STATUS_MODE="HELP"
fi

if [ "$STATUS_MODE" = "HELP" ] ; then
    desc
	echo "脚本用法： ./a200_configure.sh "
    exit 1
fi

desc



if hostname | grep 'davinci-mini' > /dev/null; then
	printRedText "please change domain and remove usb0 at first with script a200_pre_configure.sh"
	exit 5
fi

sh module/instll_toruqe.sh
sh module/modify_pbs_config.sh
sh module/add_user.sh
sh module/add_a200_dns.sh

# 添加定时任务
sh module/add_pbs_schedule_task.sh
sh module/add_device_discovery_task.sh
# 读取一下环境变量
. /etc/profile
# 通过定时任务启动PBS程序
sleep 5
if pbsnodes | grep "state = free" ; then
	printGreenText "A200 ip:$(ip) 域名: $(hostname) pbs程序已经成功运行"
	printGreenText "请前往管理节点：${manager_ip}配置该节点"
else
	printRedText "A200 ip:$(ip) 域名: $(hostname) pbs程序仍未正常启动"
fi

if [ -n $(pid_of_port 15002) ];then
	printRedText "pbs守护进程未正常启动"
fi

if [ -n $(pid_of_port 18888) ];then
	printRedText "设备发现守护进程未正常启动"
fi

printGreenText "A200 ip:$(ip) 域名: $(hostname) pbs程序已经成功部署PBS程序"
