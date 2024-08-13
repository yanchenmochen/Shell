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
printGreenText "导入功能函数，部署AI推理主服务"

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
campus_dir="../package/campus"
campus_program_path="${campus_dir}/campus"
properties_file="${campus_dir}/application-dev.properties"
if [ -e ${campus_dir}/campus.jar ]; then
	mv ${campus_dir}/campus.jar ${campus_dir}/campus
fi

if [ ! -e ${campus_program_path} ]; then
	printRedText "campus程序不存在，请正确放置"
	exit 5
fi

printGreenText "开始配置AI推理主服务"
campus_service_path="../service/campus.service"
cp ${campus_service_path} /etc/systemd/system
sed -i "s|shell-dir|"${shell_dir}"|g" /etc/systemd/system/campus.service
sed -i "s|manager_ip|"${manager_ip}"|g" /etc/systemd/system/campus.service
cat /etc/systemd/system/campus.service | grep "ExecStart"
printYellowText "准备配置application-dev.properties"
printGreenText "管理节点配置如下: "
printGreenText "manager_ip=${manager_ip}"
printGreenText "manager_domain=${manager_domain}"
printGreenText "user_name=${user_name}"
printGreenText "password=${password}"

printGreenText "数据库配置如下: "
printGreenText "db_host=${db_host}"
printGreenText "db_schema=${db_schema}"
printGreenText "db_user_name=${db_user_name}"
printGreenText "db_password=${db_password}"

printGreenText "配置项名称如下: "
printGreenText "application_name=${application_name}"

printGreenText "日志管理模块信息配置如下: "
printGreenText "remote_log_host=${remote_log_host}"

printGreenText "物联网平台配置如下: "
printGreenText "iot_ip=${iot_ip}"
printGreenText "iot_port=${iot_port}"
printGreenText "iot_user_name=${iot_user_name}"
printGreenText "iot_user_password=${iot_user_password}"


printGreenText "替换FT平台配置信息"
replace "camp.ip" "${manager_ip}" ${properties_file}
replace "camp.hostname" "${manager_domain}" ${properties_file}


printGreenText "替换数据库配置"
sleep 1
replace "db.host" "${db_host}" ${properties_file}
replace "db.schema" "${db_schema}" ${properties_file}
replace "db.username" "${db_user_name}" ${properties_file}
replace "db.password" "${db_password}" ${properties_file}

printGreenText "替换服务管理配置"
sleep 1
replace "inferEngine.ip" "${infer_engine_ip}" ${properties_file}
replace "inferEngine.userName" "${infer_engine_user_name}" ${properties_file}
replace "inferEngine.password" "${infer_engine_password}" ${properties_file}

printGreenText "替换日志配置"
sleep 1
replace "remote.log.host" "${remote_log_host}" ${properties_file}
replace "application.name" "${application_name}" ${properties_file}

printGreenText "替换物联网平台配置"
sleep 1
replace "iot.ip" "${iot_ip}" ${properties_file}
replace "iot.port" "${iot_port}" ${properties_file}
replace "iot.userName" "${iot_user_name}" ${properties_file}
replace "iot.userPassword" "${iot_user_password}" ${properties_file}


systemctl daemon-reload
echo
echo

