#!/bin/bash
#owner: songquanheng
#date: 2020年11月12日17:51:07
#脚本作用：该脚本用于批量在每个A200上部署jdk环境
#  1. 创建目录 /usr/local/java
#  2. 解压jdk-8u261-linux-arm64-vfp-hflt.tar.gz到java目录下
#  3. 配置环境变量/etc/profile
#  4. 虽然在脚本上导入了/etc/profile，但由于控制台是一个shell，脚本是它的子shell。两者不是一个环境。
#     因此需要单独在启一个shell，使得配置文件生效。

cd $(dirname $0)
pwd
. ./util/util_function.sh
. ./conf/camp.conf
desc() {
    printRedText "脚本的主要作用包括："
	printGreenText "读取ip_domain.conf中的文件内容，调用脚本deploy_java.sh在读取到的板子上部署JDK"
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
	echo "脚本用法： ./batch_deploy_java.sh"
    exit 1
fi


ip_domain_path=$(pwd)/conf/${ip_domain_config_path}
cat ${ip_domain_path} | while read line
do {
	a200_ip=$(echo $line | awk '{print $1}')
	a200_domain=$(echo $line | awk '{print $2}')
	printGreenText "为A200 ip:${a200_ip} 部署JDK环境"
	ssh root@${a200_ip} "cd /root/shell; bash deploy_java.sh"		
} &
done



