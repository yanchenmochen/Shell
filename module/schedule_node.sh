#!/bin/bash
# crontab -e  */1 * * * * su - root -c "/usr/shell/schedule_node.sh"

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

echo "$(print_date)执行node定时脚本" >> ${log_path}
node_pid=$(get_pid 8888)
if [ -z ${node_pid} ] ; then
	echo "8888为node运行端口"  >> ${log_path}
	echo "node运行状态... fail" >> ${log_path}
	echo "node运行状态: $(print_date)" >> ${log_path}
	node ${node_path}/rtsp_server/server.js > /dev/null  2>&1 &
	echo "启动命令执行完毕" >> ${log_path}
fi

