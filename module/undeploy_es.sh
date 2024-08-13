#!/bin/bash

cd $(dirname $0)

. ../util/util_function.sh
printGreenText "导入功能函数，准备卸载elasticsearch和search程序"
. ../conf/camp.conf

printGreenText "停止searchx服务"
systemctl disable searchx
systemctl stop searchx
sleep 3

printGreenText "停止elasticsearch服务"
systemctl disable elasticsearch
systemctl stop elasticsearch
sleep 3


if [ -e ${es_path}/deploy_aarch64 ] ; then
	printGreenText "删除elasticsearch和search程序"
	rm -rf /opt/deploy_aarch64
fi


if [ -e ${searchx_config_path} ]; then
	printGreenText "删除search服务配置文件"
	rm -rf /etc/search
fi


printGreenText "删除elasticsearch和search service文件"


printGreenText "移除elasticsearch用户和用户组"
if cat /etc/passwd | grep "elasticsearch" > /dev/null; then
	userdel elasticsearch

fi 
if cat /etc/group | grep "elasticsearch" > /dev/null; then
	groupdel elasticsearch
fi


cd /etc/systemd/system/
rm -rf elasticsearch.service
rm -rf searchx.service
systemctl daemon-reload

printGreenText "移除/etc/sysctl.conf中的个性化配置"
if cat /etc/sysctl.conf | grep "max_map_count" ; then
	sed -i '/max_map_count/d' /etc/sysctl.conf
	sysctl -p
fi
echo
echo
