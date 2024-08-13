#!/bin/bash

# owner: songquanheng
# date: 2020年11月17日15:03:59于文福路9号
# 警告： 需要已经处理过每个板子上多余的usb0网卡信息。这样在调用util中函数ip才能正常工作
#        该脚本需要指定a200已经正确的配置域名
# 脚本作用： 该脚本用于配置芯片的域名，并重启式域名生效。
# 脚本作用： 该脚本主要用于在/etc/hosts中添加dns信息。包括管理节点和自身
#			  同时把/etc/hosts中的127.0.1.1行修改为127.0.0.1        自身域名
#            

if [ "$1" = "-help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "--help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "help" ] ; then
        STATUS_MODE="HELP"
fi

if [ "$STATUS_MODE" = "HELP" ] ; then
    echo "脚本运行时不需要添加参数，该脚本主要作用是添加管理节点和自身的dns信息"
	echo "./change_domain  芯片IP 修改后域名"
    exit 1
fi

cd $(dirname $0)
pwd

# 导入功能模块
. ../util/util_function.sh
. ../conf/camp.conf
printGreenText "管理节点IP: ${manager_ip}"
printGreenText "管理节点域名: ${manager_domain}"

chip_ip=$(ip)
new_domain=$(hostname)

hosts_path=/etc/hosts


# 把/etc/hosts中的第二行127.0.1.1        davinci-mini修改
sed -i "s/127\.0\.1\.1.*$/127\.0\.0\.1        ${new_domain}/g" ${hosts_path}
# 在芯片/etc/hosts中添加管理节点信息
echo >> ${hosts_path}
echo "${chip_ip}        ${new_domain}" >> ${hosts_path}
echo "${manager_ip}        ${manager_domain}" >> ${hosts_path}

if grep "${manager_domain}" ${hosts_path}; then
	printGreenText "芯片ip: $(ip) 配置管理节点${manager_domain}成功"
else
	printGreenText "芯片ip: $(ip) 配置管理节点${manager_domain}失败"
	printRedText "检查${hosts_path}文件内容"
	cat ${hosts_path}
fi

if grep "${new_domain}" ${hosts_path}; then
	printGreenText "芯片ip: $(ip) 配置自身节点${new_domain}成功, 域名: ${new_domain}"
else
	printGreenText "芯片ip: $(ip) 配置自身节点${new_domain}失败"
	printRedText "检查${hosts_path}文件内容"
	cat ${hosts_path}
fi

