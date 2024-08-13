#!/bin/bash
# author：songquanheng
# 2020-04-07 10:04:39
# 脚本用途： 该脚本用于保障资源智能设备多个模块的运行状态
# crontab -e 编辑定时任务 */1 * * * * su - root -c "/usr/shell/schedule_task.sh"


# 该函数用于获取用指定语言编写、指定端口的，
# $1表示端口, 进程id处于LISTEN

cd $(dirname $0)
pwd
. ../util/util_function.sh
. ../conf/camp.conf

get_pid() {
	echo $(netstat -anop | grep ":$1" | grep -w LISTEN | tr -s " " | awk -F' ' '{print $7}' | cut -d/ -f1 | sort | uniq)
}

print_date() {
	echo $(date "+%Y-%m-%d %H:%M:%S")
}
log_dir=${daemon_log_dir}/cron
log_path=${log_dir}/cron.log
log_size=$(du -k "$log_path" | awk '{print $1}')

if [ ${log_size} -gt 10240 ] ;then 
	mv ${log_path} "/root/cron/cron-$(print_date).log"
fi

if [ ! -e ${log_dir} ]; then
	mkdir -p ${log_dir}
fi

echo "$(print_date)执行脚本" >> ${log_path}
nginx_pid=$(get_pid 8090)
if [ -z ${nginx_pid} ] ; then
	echo "nginx状态不在运行" >> ${log_path}
	echo "启动nginx: $(print_date)" >> ${log_path}
	/usr/local/nginx/sbin/nginx -c /usr/local/nginx/conf/nginx.conf
	echo "启动命令执行完毕" >> ${log_path}
fi

echo >> ${log_path}
camp_pid=$(get_pid 8080)
if [ -z ${camp_pid} ] ; then
	echo "8080端口为应用管理端口" >> ${log_path}
	echo "应用管理平台状态... fail" >> ${log_path}
	echo "启动应用管理平台 $(print_date)" >> ${log_path}
	sh ${tomcat_path}/tomcat/bin/startup.sh > /dev/null
	echo "启动命令执行完毕" >> ${log_path}
fi

echo >> ${log_path}
service_pid=$(get_pid 5000)
if [ -z ${service_pid} ] ; then
    echo "5000端口服务管理模块端口" >> ${log_path}
    echo "服务管理模块状态... fail" >> ${log_path}
    echo "启动服务管理模块 $(print_date)" >> ${log_path}
    setsid /usr/local/HSmngplt/etc/HS_flask_run.sh &
    echo "启动命令执行完毕" >> ${log_path}
fi

echo >> ${log_path}
pbs_server_pid=$(get_pid 15001)
if [ -z ${pbs_server_pid} ] ; then
	echo "15001端口为pbs_server端口" >> ${log_path}
	echo "PBS模块状态... fail" >> ${log_path}
    echo "启动PBS模块 $(print_date)" >> ${log_path}
	pbs_server	
	startPbs="true" 
	echo "启动命令执行完毕" >> ${log_path}
fi

echo >> ${log_path}
pbs_mom_pid=$(get_pid 15002)
if [ -z ${pbs_mom_pid} ] ; then
	echo "15002 15003端口为pbs_mom端口" >> ${log_path}
	echo "PBS模块状态... fail" >> ${log_path}
    echo "启动PBS模块 $(print_date)" >> ${log_path}
	pbs_mom	
	startPbs="true" 
	echo "启动命令执行完毕" >> ${log_path}
fi

echo >> ${log_path}
pbs_sched_pid=$(get_pid 15004)
if [ -z ${pbs_sched_pid} ] ; then
	echo "15004端口为pbs_sched端口" >> ${log_path}
	echo "PBS模块状态... fail" >> ${log_path}
    echo "启动PBS模块 $(print_date)" >> ${log_path}
	pbs_sched
	startPbs="true" 	
	echo "启动命令执行完毕" >> ${log_path}
fi

echo >> ${log_path}
if [ ! -z ${startPbs} ] ; then
	echo "PBS模块其他功能已执行" >> ${log_path}
    echo "执行trqauthd命令 $(print_date)" >> ${log_path}
	trqauthd	
	echo "启命令执行完毕" >> ${log_path}
fi

echo >> ${log_path}
es_pid1=$(get_pid 9200)
if [ -z ${es_pid1} ] ; then
	echo "9200端口为数据检索与外部通讯端口"  >> ${log_path}
	echo "数据检索状态... fail" >> ${log_path}
	echo "启动数据检索: $(print_date)" >> ${log_path}
	systemctl restart elasticsearch
	echo "启动命令执行完毕" >> ${log_path}
fi

echo >> ${log_path}
es_pid2=$(get_pid 9300)
if [ -z ${es_pid2} ] ; then
	echo "9300端口为数据检索节点内部通讯端口"  >> ${log_path}
	echo "数据检索内部节点通讯状态... fail" >> ${log_path}
	echo "启动数据检索: $(print_date)" >> ${log_path}
	systemctl restart elasticsearch
	echo "启动命令执行完毕" >> ${log_path}
fi

echo >> ${log_path}
search_pid=$(get_pid 9400)
if [ -z ${search_pid} ] ; then
	echo "9400端口为应用管理与检索平台"  >> ${log_path}
	echo "检索平台状态... fail" >> ${log_path}
	echo "启动检索平台: $(print_date)" >> ${log_path}
	systemctl restart searchx
	echo "启动命令执行完毕" >> ${log_path}
fi

# echo >> ${log_path}
# dm_pid=$(get_pid 5236)
# if [ -z ${dm_pid} ] ; then
	# echo "5236端口为DM数据库"  >> ${log_path}
	# echo "DM数据库状态... fail" >> ${log_path}
	# echo "启动DM数据库: $(print_date)" >> ${log_path}
	# systemctl restart DmServiceDMSERVER
	# echo "启动命令执行完毕" >> ${log_path}
# fi

echo >> ${log_path}
monitor_pid=$(get_pid 10080)
if [ -z ${monitor_pid} ] ; then
	echo "10080为运管中心端口"  >> ${log_path}
	echo "运管中心状态... fail" >> ${log_path}
	echo "启动运管中心: $(print_date)" >> ${log_path}
	bash startup_campus_monitor.sh
	echo "启动命令执行完毕" >> ${log_path}
fi

echo >> ${log_path}
node_pid=$(get_pid 8888)
if [ -z ${node_pid} ] ; then
	echo "8888为node运行端口"  >> ${log_path}
	echo "node运行状态... fail" >> ${log_path}
	echo "node运行状态: $(print_date)" >> ${log_path}
	node ${node_path}/rtsp_server/server.js > /dev/null  2>&1 &
	echo "启动命令执行完毕" >> ${log_path}
fi

echo >> ${log_path}
node1937_pid=$(get_pid 1937)
if [ -z ${node1937_pid} ] ; then
	echo "1937为node2运行端口"  >> ${log_path}
	echo "node2运行状态... fail" >> ${log_path}
	echo "node2运行状态: $(print_date)" >> ${log_path}
	node /opt/deploy_aarch64/stream/nms/app.js > /dev/null  2>&1 &
	echo "启动命令执行完毕" >> ${log_path}
fi

