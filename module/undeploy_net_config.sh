#!/bin/bash

# author：songquanheng
# 2021年7月16日11:04:56
# 脚本作用： 用于移除配置系统网络配置
cd $(dirname $0)

. ../util/util_function.sh

printGreenText "移除/etc/sysctl.conf中的个性化配置"
if cat /etc/sysctl.conf | grep "tcp_timestamps" > /dev/null ; then
	sed -i '/tcp_timestamps/d' /etc/sysctl.conf
	sysctl -p
fi