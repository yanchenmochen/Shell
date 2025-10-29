#!/bin/bash
# author: songquanheng
# date: 2020-04-22 
# desc: 该脚本用于登录芯片，并且让所有的root用户可以ssh连接。
#    注意： expect和tcl需要提前安装
cd $(dirname $0)
pwd
. ./conf/camp.conf
. ./util/util_function.sh
config_file=$1

cat $config_file | while read line
do
	remote_ip=$(echo $line | awk '{print $1}')
	remote_defalt_user=$(echo $line | awk '{print $2}')
	remote_defalt_password=$(echo $line | awk '{print $3}')
	remote_definition_password=$(echo $line | awk '{print $4}')
	$(pwd)/expect.exp $remote_ip $remote_defalt_user $remote_defalt_password $remote_definition_password
done
