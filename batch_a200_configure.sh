#!/bin/bash

# owner: songquanheng
# date: 2020年11月18日16:16:18于文福路9号
# 警告：需要已经处理过每个板子上多余的usb0网卡信息。这样在调用util中函数ip才能正常工作
# 警告：sh a200_pre_configure.sh iuni7已经成功执行过，即每个板子的域名已经成功修改。
# 警告：
# 脚本作用：批量在板子上执行a200_configure.sh脚本，作用如下 
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
	printGreenText "读取ip_domain.conf中的文件内容，调用脚本deploy_java.sh在读取到的板子上部署JDK"
} 

echo "ip_domain_config_path=${ip_domain_config_path}"
if [ -z ${ip_domain_config_path} ]; then
	printRedText "请在$(pwd)/conf/camp.conf中正确配置变量ip_domain_config_path"
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
	echo "脚本用法： ./batch_a200_configure.sh"
    exit 1
fi

ip_domain_path=$(pwd)/conf/${ip_domain_config_path}
cat ${ip_domain_path} | while read line
do {
	a200_ip=$(echo $line | awk '{print $1}')
	a200_domain=$(echo $line | awk '{print $2}')
	printGreenText "为A200 ip:${a200_ip} 部署pbs，执行a200_configure.sh脚本"
	ssh root@${a200_ip} "bash /root/shell/a200_configure.sh"		
} &
done









