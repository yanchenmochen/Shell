#!/bin/bash
# crontab -e  */1 * * * * su - root -c "/usr/shell/schedule_es.sh"

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

echo "$(print_date)执行ES定时脚本" >> ${log_path}
es_pid1=$(get_pid 9200)
if [ -z ${es_pid1} ] ; then
	echo "9200端口为数据检索与外部通讯端口"  >> ${log_path}
	echo "数据检索状态... fail" >> ${log_path}
	echo "启动数据检索: $(print_date)" >> ${log_path}
	systemctl restart elasticsearch
	echo "启动命令执行完毕" >> ${log_path}
fi

es_pid2=$(get_pid 9300)
if [ -z ${es_pid2} ] ; then
	echo "9300端口为数据检索节点内部通讯端口"  >> ${log_path}
	echo "数据检索内部节点通讯状态... fail" >> ${log_path}
	echo "启动数据检索: $(print_date)" >> ${log_path}
	systemctl restart elasticsearch
	echo "启动命令执行完毕" >> ${log_path}
fi

search_pid=$(get_pid 9400)
if [ -z ${search_pid} ] ; then
	echo "9400端口为应用管理与检索平台"  >> ${log_path}
	echo "检索平台状态... fail" >> ${log_path}
	echo "启动检索平台: $(print_date)" >> ${log_path}
	systemctl restart searchx
	echo "启动命令执行完毕" >> ${log_path}
fi

