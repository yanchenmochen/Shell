#!/bin/bash
# author：songquanheng
# 2020-04-07 10:04:39
# 脚本用途： 该脚本用于启动运管中心
# 脚本需要在module目录下执行

# 获取当前脚本所在路径
#cd $(dirname "${BASH_SOURCE[0]}") && pwd

#export JAVA_HOME=/opt/jdk1.8.0_261
#export PATH=/opt/jdk1.8.0_261/bin:$PATH

cd $(dirname $0)
pwd

. ../util/util_function.sh
. ../conf/camp.conf
printGreenText "manager_ip: ${manager_ip}"
printGreenText "user_name: ${user_name}"
printGreenText "password: ${password}"
printGreenText "log_dir=${daemon_log_dir}"
log_dir=${daemon_log_dir}/app

printGreenText "在管理节点上ip: ${manager_ip}启动运行管理守护进程"
if [ ! -e ${log_dir} ]; then
	mkdir -p ${log_dir}
fi
nohup java -jar ../package/monitor.jar --campus.host=${manager_ip} --campus.username=${user_name} --campus.password=${password} >> ${log_dir}/monitor_nohup.out 2>&1 &
