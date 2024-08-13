#!/bin/bash
#owner: songquanheng
#date: 2020年11月25日15:41:07
#脚本作用：该脚本用于在服务器上一键部署资源智能识别程序
#  调用module文件夹中脚本依次部署
#  主控程序、node、nginx、前端程序、ES、设备发现程序、运行管理中心



if [ "$1" = "-help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "--help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "help" ] ; then
        STATUS_MODE="HELP"
fi

if [ "$STATUS_MODE" = "HELP" ] ; then
    echo "请首先修改conf/camp.conf配置文件,配置相关程序要安装的目录"
    exit 1
fi
cd $(dirname $0)
pwd
. util/util_function.sh
. conf/camp.conf
printGreenText "导入功能函数"

# 判断当前用户是否为root
printYellowText "当前用户${USER}"

if test ${USER} != "root" ;then
  printGreenText "请使用root用户执行ip配置操作"
  exit 4
fi

printGreenText "管理节点配置如下: "
printGreenText "manager_ip=${manager_ip}"
printGreenText "manager_domain=${manager_domain}"
printGreenText "user_name=${user_name}"
printGreenText "password=${password}"
echo

sleep 1
printf "是否正确，请输入Y|N: "
read correct
printYellowText "请认真检查上述配置文件中的管理节点配置是否正确"
if test "${correct}" != "Y"; then
	printYellowText "请正确配置camp.conf配置中管理节点信息"
	exit 4
fi

echo
printGreenText "数据库配置如下: "
printGreenText "db_host=${db_host}"
printGreenText "db_schema=${db_schema}"
printGreenText "db_user_name=${db_user_name}"
printGreenText "db_password=${db_password}"

printYellowText "请认真检查上述配置文件中的数据库配置是否正确"
printf "是否正确，请输入Y|N: "
read correct
if test "${correct}" != "Y"; then
	printYellowText "请正确配置camp.conf配置中数据库信息"
	exit 4
fi

echo
printGreenText "服务管理配置如下: "
printGreenText "infer_engine_ip=${infer_engine_ip}"
printGreenText "infer_engine_user_name=${infer_engine_user_name}"
printGreenText "infer_engine_password=${infer_engine_password}"

printYellowText "请认真检查上述配置文件中的服务管理配置是否正确"
printf "是否正确，请输入Y|N: "
read correct
if test "${correct}" != "Y"; then
	printYellowText "请正确配置camp.conf配置中服务管理信息"
	exit 4
fi

echo
printGreenText "配置项名称如下: "
printGreenText "application_name=${application_name}"
printYellowText "请认真检查上述配置文件中的配置项配置是否正确"
echo

sleep 1
printf "是否正确，请输入Y|N: "
read correct
if test "${correct}" != "Y"; then
	printYellowText "请正确配置camp.conf配置中配置项信息"
	exit 4
fi

echo
printGreenText "日志管理模块信息配置如下: "
printGreenText "remote_log_host=${remote_log_host}"
printYellowText "请认真检查上述配置文件中的日志管理模块配置是否正确"

echo
sleep 1
printf "是否正确，请输入Y|N: "
read correct
if test "${correct}" != "Y"; then
	printYellowText "请正确配置camp.conf配置中日志管理模块信息"
	exit 4
fi

echo
printGreenText "物联网平台配置如下: "
printGreenText "iot_ip=${iot_ip}"
printGreenText "iot_port=${iot_port}"
printGreenText "iot_user_name=${iot_user_name}"
printGreenText "iot_user_password=${iot_user_password}"

printYellowText "请认真检查上述配置文件中的物联网平台配置是否正确"
echo

sleep 1
printf "是否正确，请输入Y|N: "
read correct
if test "${correct}" != "Y"; then
	printYellowText "请正确配置camp.conf配置中物联网平台配置信息"
	exit 4
fi

echo

chmod 400 conf/jmxremote*

# 配置系统网络
bash module/deploy_net_config.sh

# 调用deploy_campus.sh脚本，部署campus程序
bash module/deploy_campus_jar.sh

# 调用deploy_node.sh脚本，部署node
bash module/deploy_node.sh

# 调用deploy_nginx.sh脚本，部署nginx
bash module/deploy_nginx.sh

# 调用deploy_campus_web.sh脚本，部署web程序
bash module/deploy_campus_web.sh

# 调用deploy_es.sh脚本，部署es
# bash module/deploy_es.sh
# 部署设备发现程序
# bash module/deploy_discovery.sh

# 部署运行管理中心
bash module/deploy_monitor.sh
# 部署Java安防分析算法
bash module/deploy_algorithm.sh
#部署IOT
bash module/deploy_iot.sh

echo

printGreenText "为系统添加快捷命令"
bash module/deploy_symbolic.sh
echo
printGreenText "AI推理平台已经安装完成"

printGreenText "准备启动相关程序，较为耗时，请耐心等待..."
sleep 3
bash module/enable_module.sh
bash module/start_module.sh
echo
echo
printGreenText "AI推理平台已经启动完成..."
