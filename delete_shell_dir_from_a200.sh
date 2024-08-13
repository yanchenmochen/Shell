#!/bin/bash

# author：songquanheng
# 2020年11月23日14:55:37于文福路9号
# 该脚本用于把shell目录拷贝到多个A200芯片。
# A200芯片的信息位于conf/ip_domain.conf
# 警告：该脚本顺利执行需要已经事先配置过管理节点到各个A200的免密登录

cd $(dirname $0)
pwd
. ./util/util_function.sh
. ./conf/camp.conf

printGreenText "shell_src_dir=${shell_src_dir}"
printGreenText "shell_dest_dir=${shell_dest_dir}"
desc() {
    printRedText "脚本的主要作用包括："
	printGreenText "该脚本用于把shell目录从A200芯片上删除"
	printRedText "A200芯片的信息位于conf/ip_domain.conf"
}

echo "ip_domain_config_path=${ip_domain_config_path}"
if [ -z ${ip_domain_config_path} ]; then
	printRedText "请在$(pwd)/conf/camp.conf中正确配置变量ip_domain_config_path"
	exit 5
fi

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
	echo "脚本用法： bash detele_shell_dir_to_a200.sh"
    exit 1
fi

echo "ip_domain_path=$(pwd)/conf/${ip_domain_config_path}"
ip_domain_path=$(pwd)/conf/${ip_domain_config_path}
cat ${ip_domain_path} | while read line
do {
	a200_ip=$(echo $line | awk '{print $1}')
	# 通过免密把shell文件夹删除
	ssh root@${a200_ip} "rm -rf /root/shell"
	printGreenText "A200 ip:${a200_ip} shell文件夹已删除"
} &
done


