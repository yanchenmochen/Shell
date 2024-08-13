#!/bin/bash

# owner: songquanheng
# date: 2020年5月14日 17点07分于阿拉善百吉宾馆
# 脚本作用： 该脚本用于iSC接入模块的配置信息


if [ "$1" = "-help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "--help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "help" ] ; then
        STATUS_MODE="HELP"
fi

if [ "$STATUS_MODE" = "HELP" ] ; then
    echo "脚本运行时需要添加4个参数：一体机ip isc服务器ip isc接口接入需要的appkey isc接口接入需要的appsecret"
	echo "./change_isc_ip 一体机ip isc服务器ip isc接口接入需要的appkey isc接口接入需要的appsecret"
    exit 1
fi
cd $(dirname $0)
pwd

# 导入功能模块
. ../util/util_function.sh


if test $# -lt 4 ; then
    printRedText "脚本运行时需要添加4个参数：一体机ip isc服务器ip isc接口接入需要的appkey isc接口接入需要的appsecret"
	echo "./change_isc_ip 一体机ip isc服务器ip isc接口接入需要的appkey isc接口接入需要的appsecret"
    exit 5
fi

printGreenText "该脚本用于修改iSC接入模块的配置信息"
echo "进行参数的校验"
if test $(valid_ip "$1") != $TRUE ;then 
    printRedText "请输入合法而且真实一体机ip地址"
    printYellowText "please input -help option to get some help"
    exit 5
fi

if test $(valid_ip "$2") != $TRUE ;then 
    printRedText "请输入合法而且真实iSC服务器ip地址"
    printYellowText "please input -help option to get some help"
    exit 5
fi

if [ -n $3 ] -o [ -n $4 ]; then
	printRedText "请输入合法而且真实iSC接口接入所需要的appkey和appsecret"
	exit 5
fi

server_ip=$1
isc_ip=$2
appkey=$3
appsecret=$4

#如果server_ip的长度为非0，修改camp.ip
if [ -n "${server_ip}" ]; then
	#修改ISC接入服务IP
	echo "修改ISC接入服务IP"
	sed -i '/tomcat.ip=/c tomcat.ip='"${server_ip}" /usr/tomcat/webapps/platform/WEB-INF/classes/application.properties
fi

#打印修改后的ip
sed -n '/tomcat.ip=/p' /usr/tomcat/webapps/platform/WEB-INF/classes/application.properties

echo "进行ISC平台的IP配置的修改:"
if [ -n "${isc_ip}" ]; then
	#修改ISC接入服务IP
	sed -i '/iSecureCenter.host=/c iSecureCenter.host='"${isc_ip}" /usr/tomcat/webapps/platform/WEB-INF/classes/application.properties
fi
sed -n '/iSecureCenter.host=/p' /usr/tomcat/webapps/platform/WEB-INF/classes/application.properties

echo "配置appkey"
if [ -n "${appkey}" ]; then
	#修改ISC接入服务IP
	sed -i '/iSecureCenter.appkey=/c iSecureCenter.appkey='"${appkey}" /usr/tomcat/webapps/platform/WEB-INF/classes/application.properties
fi
sed -n '/iSecureCenter.appkey=/p' /usr/tomcat/webapps/platform/WEB-INF/classes/application.properties


echo "配置iSC接口接入所需要的appsecret"
if [ -n "${appsecret}" ]; then
	#修改ISC接入服务IP
	sed -i '/iSecureCenter.appsecret=/c iSecureCenter.appsecret='"${appsecret}" /usr/tomcat/webapps/platform/WEB-INF/classes/application.properties
fi
sed -n '/iSecureCenter.appsecret=/p' /usr/tomcat/webapps/platform/WEB-INF/classes/application.properties

echo "camp模块和iSC模块的ip成功修改"
echo "重启camp模块和iSC模块"

#ip修改成功后，启动ISC接入服务
echo "开始终止iSC接入服务"
sh /usr/tomcat/bin/shutdown.sh
# 8081端口为iSC接入模块
portkill 8081
echo "开始启动iSC接入模块"
sh /usr/tomcat/bin/startup.sh