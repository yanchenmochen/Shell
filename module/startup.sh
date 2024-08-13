#!/bin/sh
# author：songquanheng
# 2020-04-07 10:04:39
# 脚本用途： 该脚本用于设备发现程序
# 脚本需要在目录下执行

# 获取当前脚本所在路径
#cd $(dirname "${BASH_SOURCE[0]}") && pwd

#export JAVA_HOME=/opt/jdk1.8.0_261
#export PATH=/opt/jdk1.8.0_261/bin:$PATH

cd $(dirname $0)
pwd
. ../util/util_function.sh
. ../conf/camp.conf

printGreenText "daemon_log_dir=${daemon_log_dir}"

log_dir=${daemon_log_dir}/device-discovery

if [ ! -e ${log_dir} ]; then
	mkdir -p ${log_dir}
fi

if grep 'Ubuntu' /etc/lsb-release ; then
	printGreenText "在A200上启动设备发现程序，ip：$(ip)"
	nohup java -jar ../package/device-discovery.jar --type=Atlas  >> ${log_dir}/nohup.out 2>&1 &
else
	printGreenText "在FT机器上启动设备发现程序，ip：${manager_ip}"
	nohup java -jar ../package/device-discovery.jar --type=FTServer  >> ${log_dir}/nohup.out 2>&1 &
fi
