#!/bin/bash

# owner: songquanheng
# date: 2020年11月18日10:43:30于文福路9号
# 警告： 需要已经处理过每个板子上多余的usb0网卡信息。这样在调用util中函数ip才能正常工作。并且芯片IP已经正确配置
# 脚本作用： 该脚本预先对A200进行处理，主要作用是移除虚拟网卡usb0的信息，并且修改域名,该脚本会重启a200

if [ "$1" = "-help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "--help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "help" ] ; then
        STATUS_MODE="HELP"
fi

if [ "$STATUS_MODE" = "HELP" ] ; then
	echo "脚本主要作用: 该脚本预先对A200进行处理，主要作用是移除虚拟网卡usb0的信息，并且修改域名,该脚本会重启a200"
    echo "脚本运行时需要添加1个参数：修改后域名"
	echo "./a200_pre_configure 修改后域名"
    exit 1
fi

cd $(dirname $0)
pwd
. ./conf/camp.conf
. ./util/util_function.sh

if [ -z $1 ] ; then
	printRedText "请输入合法而且有效的修改后域名"
	exit 5
fi

new_a200_domain=$1


if grep "usb0" /etc/network/interfaces > /dev/null; then
	printGreenText "移除虚拟网卡usb0的信息"
	chmod +x ./module/remove_usb0.sh
	./module/remove_usb0.sh
else
	printYellowText "在配置文件/etc/network/interfaces中不包含虚拟网卡的信息"
fi
chmod +x ./module/change_domain.sh
./module/change_domain.sh ${new_a200_domain}