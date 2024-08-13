#!/bin/bash
# author: songquanheng
# date: 2020-04-22 
# desc: 该脚本用于批量推送root公钥到指定的ip列表
#    注意： expect和tcl需要提前安装	

cd $(dirname $0)
pwd
. ./conf/camp.conf
. ./util/util_function.sh

ip_domain_path=$(pwd)/conf/${ip_domain_config_path}
cat ${ip_domain_path} | while read line
do
	a200_ip=$(echo $line | awk '{print $1}')
	bash push_pub.sh $a200_ip
done

