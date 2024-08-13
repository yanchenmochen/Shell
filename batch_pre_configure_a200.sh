#!/bin/bash

# owner: songquanheng
# date: 2020年11月18日20:37:17于文福路9号
# 警告：在该脚本执行时，应已经配置过从管理节点到A200的免密登录
# 脚本作用： 
# 1. 该脚本用于批量的把ip_domain_config_file所对应的配置文件中的所有ip和域名的对进行处理。
#    把每个板子修改成对应的域名。需要删除每个A200上的虚拟网卡usb0网卡信息。

cd $(dirname $0)
pwd
. ./util/util_function.sh
. ./conf/camp.conf
desc() {
    printRedText "脚本的主要作用包括："
	printGreenText "调用脚本a200_pre_configure.sh批量处理ip_domain.conf中的文件内容"
}

echo "ip_domain_config_path=${ip_domain_config_path}"
if [ -z ${ip_domain_config_path} ]; then
	printRedText "请在$(pwd)/conf/camp.conf中正确配置变量ip_domain_config_path"
	exit 5
fi

if [ `whoami` != 'root' ]; then
	echo "please login in with root user"
	exit 5
fi

if [ "$1" = "-help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "--help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "help" ] ; then
        STATUS_MODE="HELP"
fi

if [ "$STATUS_MODE" = "HELP" ] ; then
    desc
	echo "脚本用法： ./batch_pre_configure_a200.sh"
    exit 1
fi

desc
ip_domain_path=$(pwd)/conf/${ip_domain_config_path}
echo "ip_domain_path=${ip_domain_path}"
cat ${ip_domain_path} | while read line
do {
	a200_ip=$(echo $line | awk '{print $1}')
	a200_domain=$(echo $line | awk '{print $2}')
	printGreenText "为A200 ip:${a200_ip} 进行预处理，移除虚拟网卡usb0并修改成域名${a200_domain}"
	ssh root@${a200_ip} "cd /root/shell; bash a200_pre_configure.sh ${a200_domain}"	
} &
done
wait
printGreenText "处理完成，请等待所有板子执行成功"
sleep 10

cat ${ip_domain_path} | while read line
do {
	a200_ip=$(echo $line | awk '{print $1}')
	a200_domain=$(echo $line | awk '{print $2}')
	ssh root@${a200_ip} 'echo "a200_ip: $(ip) 域名： $(hostname); "'	
} &
done
