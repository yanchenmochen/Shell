#!/bin/bash

# 判断是否经由软连接命令执行
real_work_dir=`find_soft_link_real_path campus-update`

if [[ ${real_work_dir} != "" && ${real_work_dir} != `pwd` ]];then
    cd ${real_work_dir}
fi


. ../conf/camp.conf
. ../util/util_function.sh

bash menu-campus-config.sh
conf_file=../conf/camp.conf

campus_dir="../package/campus"
campus_properties_file="${campus_dir}/application-dev.properties"
campus_service_file=/etc/systemd/system/campus.service

algorithm_dir="../package/algorithm"
algorithm_properties_file="${algorithm_dir}/application-dev.properties"
algorithm_service_file=/etc/systemd/system/campus-algorithm.service

monitor_dir="../package/monitor"
monitor_properties_file="${monitor_dir}/application-dev.properties"
monitor_service_file=/etc/systemd/system/campus-monitor.service

while true
do
	read -p "请您输入要更新或者查看的配置(输入help|h|H获取帮助,ctrl+c终止更新, q|quit用于写入重启):" model_name
	case $model_name in
		help|h|H)
            bash menu-campus-config.sh
            ;;
        g|gl)
            printGreenText "管理节点信息如下："
            printGreenText "管理节点IP：${manager_ip}"
            read -p "请输入新的管理节点IP信息(ip)：" new_manager_ip
            replace "manager_ip" ${new_manager_ip} ${conf_file}
            replace "camp.ip" ${new_manager_ip} ${campus_properties_file}
            replace "camp.ip" ${new_manager_ip} ${algorithm_properties_file}
            replace "campus.host" ${new_manager_ip} ${monitor_properties_file}
            replace_hostname_in_service "-Djava.rmi.server.hostname" ${new_manager_ip} ${campus_service_file}
            replace_hostname_in_service "-Djava.rmi.server.hostname" ${new_manager_ip} ${algorithm_service_file}
            replace_hostname_in_service "-Djava.rmi.server.hostname" ${new_manager_ip} ${monitor_service_file}
            ;;
        f|fx)
            printGreenText "分析节点信息如下："
            printGreenText "分析节点IP: ${infer_engine_ip}"
            printGreenText "分析节点用户名: ${infer_engine_user_name}"
            printGreenText "分析节点密码: ${infer_engine_password}"

            read -p "请输入新的分析节点IP信息(ip)：" analysis_ip
            read -p "请输入新的分析节点用户名：" new_analysis_user_name
            read -p "请输入新的分析节点密码：" new_analysis_password
            # 修改camp.conf中对应的配置内容
            replace "infer_engine_ip" "${analysis_ip}" ${conf_file}
            replace "infer_engine_user_name" "${new_analysis_user_name}" ${conf_file}
            replace "infer_engine_password" "${new_analysis_password}" ${conf_file}
            # 修改程序安装包的配置文件中对应的内容
            replace "inferEngine.ip" "${analysis_ip}" ${campus_properties_file}
            replace "inferEngine.userName" "${new_analysis_user_name}" ${campus_properties_file}
            replace "inferEngine.password" "${new_analysis_password}" ${campus_properties_file}

            replace "pbs.host" "${analysis_ip}" ${monitor_properties_file}
            replace "service.host" "${analysis_ip}" ${monitor_properties_file}

            replace "inferEngine.ip" "${analysis_ip}" ${algorithm_properties_file}


            ;;
        iot|IOT|i|I)
            printGreenText "物联网平台配置如下: "
            printGreenText "iot_ip=${iot_ip}"
            printGreenText "iot_port=${iot_port}"
            printGreenText "iot_user_name=${iot_user_name}"
            printGreenText "iot_user_password=${iot_user_password}"

            read -p "请输入新的物联网平台配置IP信息(ip)：" iot_ip
            read -p "请输入新的物联网平台配置端口信息(port)：" iot_port
            read -p "请输入新的物联网平台用户名(userName): " iot_user_name
            read -p "请输入新的物联网平台密码(password): " iot_password

            replace "iot_ip" "${iot_ip}" ${conf_file}
            replace "iot_port" "${iot_port}" ${conf_file}
            replace "iot_user_name" ${iot_user_name} ${conf_file}
            replace "iot_user_password" ${iot_password} ${conf_file}

            replace "iot.ip" "${iot_ip}" ${campus_properties_file}
            replace "iot.port" "${iot_port}" ${campus_properties_file}
            replace "iot.userName" ${iot_user_name} ${campus_properties_file}
            replace "iot.userPassword" ${iot_password} ${campus_properties_file}
            ;;

        db|DB|d|D)
            printGreenText "数据库配置如下: "
            printGreenText "db_host=${db_host}"
            printGreenText "db_schema=${db_schema}"
            printGreenText "db_user_name=${db_user_name}"
            printGreenText "db_password=${db_password}"


            read -p "请输入新的数据库配置ip信息(ip)：" db_ip
            read -p "请输入新的数据库配置port信息(port)：" db_port
            replace "db_host" "${db_ip}:${db_port}" ${conf_file}

            replace "db.host" "${db_ip}:${db_port}" ${campus_properties_file}
            replace "db.host" "${db_ip}:${db_port}" ${algorithm_properties_file}
            replace "db.host" "${db_ip}:${db_port}" ${monitor_properties_file}
            ;;

		log|l|L)
            printGreenText "日志管理平台信息配置如下: "
            printGreenText "remote_log_host=${remote_log_host}"

            read -p "请输入新的日志管理平台ip信息(ip)：" log_ip
            read -p "请输入新的日志管理平台port信息(port)：" log_port

            # 修改camp.conf中对应的配置内容
            replace "remote_log_host" "${log_ip}:${log_port}" ${conf_file}
            # 修改程序安装包的配置文件中对应的内容
            replace "remote.log.host" "${log_ip}:${log_port}" ${campus_properties_file}

            ;;

		j|health|hea|HEA)
            printGreenText "健康管理配置如下: "
            printGreenText "health_host=${health_host}"

            read -p "请输入新的健康管理平台ip信息(ip)：" hea_ip
            read -p "请输入新的健康管理平台port信息(port)：" hea_port

            replace "health_host" "${hea_ip}:${hea_port}" ${conf_file}
            replace "health.host" "${hea_ip}:${hea_port}" ${monitor_properties_file}

            ;;
		quit|q)
            printGreenText "写入新的配置，重新启动AI智能推理程序, 请等待"
            bash ./start_module.sh
            exit

		    ;;
	esac
done