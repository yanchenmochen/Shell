#!/bin/bash
#查看推理服务各组件涉及的配置内容


# 判断是否经由软连接命令执行
real_work_dir=`find_soft_link_real_path campus-config`
echo "campus-config脚本所在真实目录---" ${real_work_dir}
if [[ ${real_work_dir} != "" && ${real_work_dir} != `pwd` ]];then
    cd ${real_work_dir}
fi

. ../conf/camp.conf
. ../util/util_function.sh

bash menu-campus-config.sh
while true
do
	read -p "请您输入要查看的配置:" model_name
	case $model_name in
		help|h|H)
		bash menu-campus-config.sh
		;;
		g|gl)
		printGreenText "管理节点信息如下："
        printGreenText "管理节点IP：${manager_ip}"
		;;
		f|fx)
		printGreenText "分析节点IP=$infer_engine_ip"
		printGreenText "分析节点用户名: ${infer_engine_user_name}"
        printGreenText "分析节点密码: ${infer_engine_password}"
		;;
		log|l|L)
        printGreenText "日志管理模块信息配置如下: "
        printGreenText "remote_log_host=${remote_log_host}"
		;;
		db|DB|d|D)
		printGreenText "数据库配置如下: "
        printGreenText "db_host=${db_host}"
        printGreenText "db_schema=${db_schema}"
        printGreenText "db_user_name=${db_user_name}"
        printGreenText "db_password=${db_password}"
		;;
		iot|IOT|i|I)
		printGreenText "物联网平台配置如下: "
        printGreenText "iot_ip=${iot_ip}"
        printGreenText "iot_port=${iot_port}"
        printGreenText "iot_user_name=${iot_user_name}"
        printGreenText "iot_user_password=${iot_user_password}"
		;;
		j|health|hea|HEA)
		printGreenText "健康管理配置如下: "
        printGreenText "health_host=${health_host}"
		;;
		quit|q)
		exit
		;;
	esac
done
