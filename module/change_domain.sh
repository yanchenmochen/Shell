#!/bin/bash

# owner: songquanheng
# date: 2020年11月17日15:03:59于文福路9号
# 警告： 需要已经处理过每个板子上多余的usb0网卡信息。这样在调用util中函数ip才能正常工作。并且芯片IP已经正确配置
# 脚本作用： 该脚本用于配置芯片的域名，并重启式域名生效。
# 脚本作用： 该脚本执行会修改/etc/hostname，清空其中的内容，并重启A200芯片
#            同时会修改/etc/hosts 把第二行替换为127.0.0.1        修改后域名


if [ "$1" = "-help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "--help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "help" ] ; then
        STATUS_MODE="HELP"
fi

if [ "$STATUS_MODE" = "HELP" ] ; then
    echo "脚本运行时需要添加1个参数：修改后域名"
	echo "./change_domain 修改后域名"
    exit 1
fi

cd $(dirname $0)
pwd

. ../util/util_function.sh
. ../conf/camp.conf

if test $# -lt 1 ; then
    printRedText "脚本运行时需要添加1个参数：修改后域名"
	echo "./chage_domain 修改后域名"
    exit 5
fi

printGreenText "该脚本用于修改指定IP: $(ip)的芯片，修改域名后重启"

if [ -z $1 ] ; then
	printRedText "请输入合法而且有效的修改后域名"
	exit 5
fi
new_a200_domain=$1

hostname_path=/etc/hostname

cat /dev/null>${hostname_path}
echo "${new_a200_domain}" >> ${hostname_path}
if cat ${hostname_path} | grep ${new_a200_domain} > /dev/null ;then 
	printGreenText "芯片$(ip) 域名修改为${new_a200_domain},重启生效"
	reboot
else
	printRedText "芯片$(ip) 域名修改为${new_a200_domain},重启生效"
	printGreenText "请检查文件"
	cat -v ${hostname_path}
fi


