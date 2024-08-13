#!/bin/bash

# owner: songquanheng
# date: 2020年11月17日16:13:48
# 警告： 需要已经处理过每个板子上多余的usb0网卡信息。这样在调用util中函数ip才能正常工作
# 脚本作用： 该脚本用于配置芯片的torque中的配置，主要修改的是server_name文件和config文件
# 添加管理节点信息

cd $(dirname $0)
pwd
. ../conf/camp.conf
. ../util/util_function.sh

if grep "HPCUser" /etc/passwd > /dev/null; then
	echo "HPCUser用户已创建"
	exit 5
fi
printGreenText "为$(ip)添加用户HPCUser"

useradd HPCUser -o  -u 0 -g 0 -m  -p 123456 -s /bin/bash
echo HPCUser:123456 | chpasswd

printRedText "HPCUser用户已创建"
cat /etc/passwd | grep HPCUser
