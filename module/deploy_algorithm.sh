#!/bin/bash
# author：songquanheng
# 2020-11-24 10:04:39
# 脚本用途： 该脚本用于部署安防分析算法
# 1、从camp.conf获取后端程序安装路径
# 2、把campus-algorithm.service拷贝到/etc/systemd/system
# 3、正确配置campus-algorithm.service
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
algorithm_dir="../package/algorithm"
if [ -e ${algorithm_dir}/campus-algorithm.jar ]; then
	mv ${algorithm_dir}/campus-algorithm.jar ${algorithm_dir}/campus-algorithm
fi

algorithm_program_path="${algorithm_dir}/campus-algorithm"


if [ ! -e ${algorithm_program_path} ]; then
	printRedText "安防算法程序不存在，请放置"
	exit 5
fi



printGreenText "开始配置安防分析算法"
algorithm_service_path="../service/campus-algorithm.service"
cp ${algorithm_service_path} /etc/systemd/system
sed -i "s|shell-dir|"${shell_dir}"|g" /etc/systemd/system/campus-algorithm.service
sed -i "s|manager_ip|"${manager_ip}"|g" /etc/systemd/system/campus-algorithm.service


cat /etc/systemd/system/campus-algorithm.service | grep "ExecStart"


systemctl daemon-reload

printYellowText "准备配置application-dev.properties"

properties_file="${algorithm_dir}/application-dev.properties"
printGreenText "替换FT平台配置信息"
sleep 1
replace "camp.ip" "${manager_ip}" ${properties_file}

printGreenText "替换数据库配置"
sleep 1
replace "db.host" "${db_host}" ${properties_file}
replace "db.schema" "${db_schema}" ${properties_file}
replace "db.username" "${db_user_name}" ${properties_file}
replace "db.password" "${db_password}" ${properties_file}

printGreenText "替换服务管理配置"
sleep 1
replace "inferEngine.ip" "${infer_engine_ip}" ${properties_file}

echo
echo

systemctl daemon-reload
