#!/bin/bash
# crontab -e  */1 * * * * su - root -c "/usr/shell/schedule_service.sh"

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

echo "$(print_date)执行service服务管理定时脚本" >> ${log_path}
service_pid=$(get_pid 5000)
if [ -z ${service_pid} ] ; then
    echo "5000端口服务管理模块端口" >> ${log_path}
    echo "服务管理模块状态... fail" >> ${log_path}
    echo "启动服务管理模块 $(print_date)" >> ${log_path}
    setsid /usr/local/HSmngplt/etc/HS_flask_run.sh &
    echo "启动命令执行完毕" >> ${log_path}
fi

