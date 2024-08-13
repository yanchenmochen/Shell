#!/bin/bash

# owner: songquanheng
# date: 2020年5月14日 17点07分于阿拉善百吉宾馆
# 脚本作用： 该脚本用于智能分析模块的ip


if [ "$1" = "-help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "--help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "help" ] ; then
        STATUS_MODE="HELP"
fi

if [ "$STATUS_MODE" = "HELP" ] ; then

    echo "脚本运行时需要添加1个参数：一体机ip"
	echo "./change_campus_ip 一体机ip"
    exit 1
fi
cd $(dirname $0)
pwd
# 导入功能模块
. ../util/util_function.sh


if test $# -ne 1 ; then
    printRedText "脚本运行时需要添加1个参数：一体机ip"
	echo "./change_campus_ip 一体机ip"
    exit 5
fi

printGreenText "该脚本用于修改智能分析模块的配置信息"
echo "进行参数的校验"
if test $(valid_ip "$1") != $TRUE ;then 
    printRedText "请输入合法而且真实一体机ip地址"
    printYellowText "please input -help option to get some help"
    exit 5
fi

server_ip=$1

#如果server_ip的长度为非0，修改camp.ip
if [ -n "${server_ip}" ]; then
	#修改资源智能识别服务IP
	echo "修改资源智能识别服务IP"
	sed -i '/camp.ip=/c camp.ip='"${server_ip}" /opt/tomcat/webapps/ROOT/WEB-INF/classes/config/application-dev.properties
fi

#打印修改后的ip
sed -n '/camp.ip=/p' /opt/tomcat/webapps/ROOT/WEB-INF/classes/config/application-dev.properties

echo "智能分析平台ip成功修改"
echo "重启camp模块和iSC模块"

#ip修改成功后，启动智能分析服务
echo "开始重启智能分析服务"
echo "终止智能分析服务"
sh /opt/tomcat/bin/shutdown.sh
# 8080端口为智能分析设备
portkill 8080
echo "启动智能分析服务"
sh /opt/tomcat/bin/startup.sh
