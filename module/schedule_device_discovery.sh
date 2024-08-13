#!/bin/bash
# author：songquanheng
# 2020-04-07 10:04:39
# 脚本用途： 该脚本定时维护A200上设备发现的程序


# 该函数用于获取用指定语言编写、指定端口的，
# $1表示端口, 进程id处于LISTEN
get_pid() {
	echo $(netstat -anop | grep ":$1" | grep -w LISTEN | tr -s " " | awk -F' ' '{print $7}' | cut -d/ -f1 | sort | uniq)
}


print_date() {
	echo $(date "+%Y-%m-%d %H:%M:%S")
}



cd $(dirname $0)
pwd

. ../util/util_function.sh
. ../conf/camp.conf

printGreenText "log_dir=${daemon_log_dir}"
log_dir=${daemon_log_dir}

install_base_dir=${log_dir}/device-discovery
if [ ! -e ${install_base_dir}/logs/cron/ ]; then
	mkdir -p ${install_base_dir}/logs/cron/
fi

log_path=${install_base_dir}/logs/cron/cron.log
if [ ! -f ${log_path}  ] ;then 
	touch ${log_path} 
fi

log_size=$(du -k "$log_path" | awk '{print $1}')

if [ ${log_size} -gt 10 ] ;then 
	mv ${log_path} "${install_base_dir}/logs/cron/cron-$(print_date).log"
fi

echo >> ${log_path}
echo >> ${log_path}
echo "正在执行定时任务: $(print_date)" >> ${log_path}

echo >> ${log_path}
node_pid=$(get_pid 18888)
if [ -z ${node_pid} ] ; then
	echo "18888端口为发现设备端口" >> ${log_path}
	echo "发现设备... fail" >> ${log_path}
	echo "启动设备发现设备:  $(print_date)" >> ${log_path}
	sh $(pwd)/startup_campus_discovery.sh
	echo "启动发现设备成功:  $(print_date)" >> ${log_path}
	
fi


