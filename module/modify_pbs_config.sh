#!/bin/bash

# owner: songquanheng
# date: 2020年11月17日15:55:59
# 警告： 需要已经处理过每个板子上多余的usb0网卡信息。这样在调用util中函数ip才能正常工作
# 脚本作用： 该脚本用于配置芯片的torque中的配置，主要修改的是server_name文件和config文件
# 添加管理节点信息

cd $(dirname $0)
pwd
. ../conf/camp.conf
. ../util/util_function.sh

cd /var/spool/torque
echo "管理节点域名: ${manager_domain}"

cat /dev/null > server_name; 
echo "${manager_domain}" > server_name

pbs_config_path=/var/spool/torque/mom_priv/config

if [ ! -e ${pbs_config_path} ]; then
    printYellowText "第一次配置PBS，需要为A200添加config文件"
    cd /var/spool/torque/;
	touch mom_priv/config; 
	echo '$logevent' 0x1ff >> mom_priv/config ;
	echo "\$pbsserver ${manager_domain}" >> mom_priv/config;
	echo '$check_poll_time 1' >> mom_priv/config;
	echo '$status_update_time 1' >> mom_priv/config
fi

printYellowText "$(ip)芯片上配置PBSserver_name和config文件成功"