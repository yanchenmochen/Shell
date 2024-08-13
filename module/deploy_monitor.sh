#!/bin/bash
# author：songquanheng
# 2020-11-24 10:04:39
# 脚本用途： 该脚本用于部署运行管理中心
# 1、从camp.conf获取后端程序安装路径
# 2、把campus-monitor.service拷贝到/etc/s ystemd/system
# 3、正确配置campus-monitor.service
# 4、正确配置application-dev.properties

cd $(dirname $0)
current_dir=$(pwd)
. ../util/util_function.sh
printGreenText "导入功能函数，部署运行管理中心"

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
monitor_dir="../package/monitor"
monitor_program_path="${monitor_dir}/campus-monitor"

if [ -e "${monitor_dir}/campus-monitor.jar" ]; then
	mv ${monitor_dir}/campus-monitor.jar ${monitor_dir}/campus-monitor
fi
if [ ! -e ${monitor_program_path} ]; then
	printRedText "运行管理中心程序不存在，请正确放置"
	exit 5
fi

printGreenText "开始配置运行管理中心"
monitor_service_path="../service/campus-monitor.service"
cp ${monitor_service_path} /etc/systemd/system
sed -i "s|shell-dir|"${shell_dir}"|g" /etc/systemd/system/campus-monitor.service
sed -i "s|manager_ip|"${manager_ip}"|g" /etc/systemd/system/campus-monitor.service
cat /etc/systemd/system/campus-monitor.service | grep "ExecStart"

properties_file="${monitor_dir}/application-dev.properties"

printGreenText "为运行管理中心配置FT平台信息"
sleep 1
replace "campus.host" "${manager_ip}" ${properties_file}
replace "campus.username" "${user_name}" ${properties_file}
replace "campus.password" "${password}" ${properties_file}

printGreenText "为运行管理中心配置服务管理信息"
sleep 1
replace "pbs.host" "${infer_engine_ip}" ${properties_file}
replace "service.host" "${infer_engine_ip}" ${properties_file}

printGreenText "为运行管理中心配置健康管理平台"
sleep 1
replace "application.name" "${application_name}" ${properties_file}
replace "application.version" "${application_version}" ${properties_file}
replace "health.host" "${health_host}" ${properties_file}

printGreenText "为运行管理中心配置数据库信息"
sleep 1
replace "db.host" "${db_host}" ${properties_file}
replace "db.schema" "${db_schema}" ${properties_file}
replace "db.username" "${db_user_name}" ${properties_file}
replace "db.password" "${db_password}" ${properties_file}
systemctl daemon-reload
echo
echo 

