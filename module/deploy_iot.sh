#!/bin/bash
# author：songquanheng
# 2020年12月30日11:20:07
# 脚本用途： 该脚本用于部署AI主服务
# 1、从camp.conf获取后端程序安装路径
# 2、正确配置文件application-dev.properties
# 2、把campus.service拷贝到/etc/systemd/system
# 3、正确配置campus.service


cd $(dirname $0)
current_dir=$(pwd)
. ../util/util_function.sh
printGreenText "导入功能函数，部署AI推理IOT服务"

# 导入配置文件，获取安装路径
. ../conf/camp.conf
cd ..
shell_dir=$(pwd)
printGreenText "shell根目录为: ${shell_dir}"
cd ${current_dir}

if grep 'Kylin' /etc/lsb-release > /dev/null ; then
	printGreenText "当前系统为Kylin操作系统"
else
	printGreenText "当前系统为A200加速卡，系统为Ubuntu"
fi
sleep 2
campus_dir="../package/iot"
campus_program_path="${campus_dir}/campus-iot"
properties_file="${campus_dir}/application-dev.properties"
if [ -e ${campus_dir}/campus-iot.jar ]; then
	mv ${campus_dir}/campus-iot.jar ${campus_dir}/campus-iot
fi

if [ ! -e ${campus_program_path} ]; then
	printRedText "campus-iot程序不存在，请正确放置"
	exit 5
fi

printGreenText "开始配置AI推理IOT服务"
campus_service_path="../service/campus-iot.service"
cp ${campus_service_path} /etc/systemd/system
sed -i "s|shell-dir|"${shell_dir}"|g" /etc/systemd/system/campus-iot.service
sed -i "s|manager_ip|"${manager_ip}"|g" /etc/systemd/system/campus-iot.service
cat /etc/systemd/system/campus.service | grep "ExecStart"

systemctl daemon-reload
echo
echo

