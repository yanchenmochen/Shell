#!/bin/bash

# author：songquanheng
# 2021年7月16日11:04:56
# 脚本作用： 用于配置系统网络配置
# NAT过来的数据包，又因为时间戳可能不是顺序的，导致服务器认为包不可信而丢弃。
# 解决Atlas 200 数据上报Http Post Failed

cd $(dirname $0)

. ../util/util_function.sh
sysctl_conf='/etc/sysctl.conf'


# 在系统配置文件中找不到vm.max_map_count
if cat ${sysctl_conf} | grep "net.ipv4.tcp_timestamps" > /dev/null; then
	printGreenText "在配置文件中已经存在net.ipv4.tcp_timestamps"
else
	printGreenText "在配置文件中添加net.ipv4.tcp_timestamps值"
	echo 'net.ipv4.tcp_timestamps=0'>>${sysctl_conf}

fi
sysctl -p