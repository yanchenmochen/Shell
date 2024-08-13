#!/bin/bash
# author：songquanheng
# 2020-11-24 10:04:39
# 脚本用途： 该脚本用于部署设备发现程序
# 1、从camp.conf获取后端程序安装路径
# 2、把campus-discovery.service拷贝到/etc/systemd/system
# 3、正确配置campus-discovery.service
# 4、正确配置application-dev.properties文件。

cd $(dirname $0)
current_dir=$(pwd)
. ../util/util_function.sh
printGreenText "导入功能函数，部署设备发现程序"

# 导入配置文件，获取安装路径
. ../conf/camp.conf
cd ..
shell_dir=$(pwd)
cd ${current_dir}

if grep 'Kylin' /etc/lsb-release > /dev/null ; then
	printGreenText "当前系统为Kylin操作系统"
else
	printGreenText "当前系统为A200加速卡，系统为Ubuntu"
fi
sleep 2
discovery_dir="../package/discovery"
if [ -e ${discovery_dir}/campus-discovery.jar ]; then
	mv ${discovery_dir}/campus-discovery.jar ${discovery_dir}/campus-discovery
fi

discovery_program_path="${discovery_dir}/campus-discovery"
properties_file="${discovery_dir}/application-dev.properties"

if [ ! -e ${discovery_program_path} ]; then
	printRedText "设备发现程序不存在，请放置"
	exit 5
fi


printGreenText "开始配置设备发现程序"
discovery_service_path="../service/campus-discovery.service"
cp ${discovery_service_path} /etc/systemd/system
sed -i "s|shell-dir|"${shell_dir}"|g" /etc/systemd/system/campus-discovery.service
sed -i "s|manager_ip|"${manager_ip}"|g" /etc/systemd/system/campus-discovery.service


if grep 'Kylin' /etc/lsb-release > /dev/null; then
	replace "type" "FTServer" ${properties_file}
	replace "network-card-name" "${network_card_name}" ${properties_file}
else
	replace "type" "Atlas" ${properties_file}
	replace "network-card-name" "eth0" ${properties_file}
fi


cat /etc/systemd/system/campus-discovery.service | grep "ExecStart"
systemctl daemon-reload
