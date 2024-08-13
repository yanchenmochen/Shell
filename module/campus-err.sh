#!/bin/bash
# author：songquanheng
# 2020-04-07 10:04:39
# 脚本用途： 该脚本用于查看各模块的错误日志
function menu() {
cat<<-EOF
 h. 帮助help|h
 c. 主控节点campus|c
 a. 安防分析算法campus-algorithm|a
 m. 运行管理中心campus-monitor|m
 q. 退出程序quit|q
EOF
}

menu

while true
do
	read -p "请您输入操作或者模块名称查看日志:" model_name
	case $model_name in
		help|h)
		menu
		;;
		campus|c)
		tail -f -n 300 /var/log/campus/error.log
		;;
		algorithm|a)
		tail -f -n 300 /var/log/algorithm/error.log
		;;
		monitor|m)
		tail -f -n 300 /var/log/monitor/error.log
		;;
		quit|q)
		exit
		;;
	esac
done
