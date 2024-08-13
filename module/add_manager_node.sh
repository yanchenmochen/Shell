#!/bin/bash

# owner: songquanheng
# date: 2020年11月17日17:05:48
# 警告： 
# 脚本作用： 该脚本用于为管理节点添加已配置的A200节点信息
# 			该脚本会修改nodes文件和/etc/hosts

if [ "$1" = "-help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "--help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "help" ] ; then
        STATUS_MODE="HELP"
fi


if [ "$STATUS_MODE" = "HELP" ] ; then
    echo "脚本运行时需要添加2个参数：芯片IP 芯片域名"
	echo "./add_manager_node.sh ip domain"
	exit 1
fi

cd $(dirname $0)
pwd
. ../util/util_function.sh
. ../conf/camp.conf

if test $# -lt 2 ; then
    printRedText "脚本运行时需要添加2个参数：已成功配置的A200IP 对应的域名"
	echo "./add_manager_node.sh A200IP A200域名"
    exit 5
fi
a200_ip=$1
a200_domain=$2
cd $(dirname $0)

nodes_path=/var/spool/torque/server_priv/nodes
echo "${a200_domain} np=1">>${nodes_path}
hosts_path=/etc/hosts
echo "${a200_ip} ${a200_domain}">>${hosts_path}

printGreenText "重启管理节点ip: ${manager_ip} 域名：${manager_domain}pbs服务"
# 重启pbs服务
sh pbs restart

pbsnodes
if pbsnodes | grep ${a200_domain} > /dev/null; then
	printGreenText "芯片A200ip: ${a200_ip} 域名: ${a200_domain}已由pbs服务管理"
else
	printRedText "芯片A200ip: ${a200_ip} 域名: ${a200_domain}仍未由pbs服务管理"
	
	printGreenText "检查${nodes_path}中内容"
	cat ${nodes_path}
	printGreenText "检查${hosts_path}中内容"
	cat ${hosts_path}
fi

