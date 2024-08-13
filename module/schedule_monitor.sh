#!/bin/bash
# crontab -e  */1 * * * * su - root -c "/usr/shell/schedule_monitor.sh"

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

echo "$(print_date)执行运管中心定时脚本" >> ${log_path}
monitor_pid=$(get_pid 10080)
if [ -z ${monitor_pid} ] ; then
	echo "10080为运管中心端口"  >> ${log_path}
	echo "运管中心状态... fail" >> ${log_path}
	echo "启动运管中心: $(print_date)" >> ${log_path}
	bash startup_campust_monitor.sh
	echo "启动命令执行完毕" >> ${log_path}
fi

