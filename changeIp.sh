#!/bin/bash
#owner: songquanheng
#date: 2020年5月14日 14点11分
#脚本作用：该脚本用于一键修改401项目相关的服务ip，其中修改的内容包括
#  1. 模块管理
#     1. camp项目的ip
#     2. camp项目中数据库所在的ip，该ip与camp的ip相同
#     3. hikinfer.ip: 服务管理所在的ip
#     4. network.host: elasticsearhsearch的ip
#     5. es.network: search 检索平台的ip地址
#  2. iSC相关
#     tomcat.ip: isc取流服务所在的服务器ip
#     1. isc_ip: isc服务器所在的ip
#     2. appkey: isc接口接入时需要传入的appkey, 在运行管理中心-状态监控-API网关-参数配置-API管理
#     3. appsecret: isc接口接入时需要传入的appsecret

#  2. 推理服务模块
#     1. 该部分对应文件/hikmars3/system/logs/config.json，直接删掉，然后添加即可

if [ "$1" = "-help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "--help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "help" ] ; then
        STATUS_MODE="HELP"
fi

if [ "$STATUS_MODE" = "HELP" ] ; then
    echo "请首先修改conf/camp.conf配置文件，修改一体机ip、isc服务ip，appkey和appsecret，然后执行ip修改操作"
    exit 1
fi
cd $(dirname $0)
pwd
. ./util/util_function.sh
printGreenText "导入功能函数"

# 判断当前用户是否为root
printYellowText "当前用户${USER}"

if test ${USER} != "root" ;then
  printGreenText "请使用root用户执行ip配置操作"
  exit 4
fi

source ./conf/401.conf 

echo "server_ip=${server_ip}"
echo "isc_ip=${isc_ip}"
echo "appkey=${appkey}"
echo "appsecret=${appsecret}"

printGreenText "开始配置iSC接入模块"
./module/change_isc_ip.sh ${server_ip} ${isc_ip} ${appkey} ${appsecret}

printGreenText "开始配置智能分析模块"
./module/change_campus_ip.sh ${server_ip}

printGreenText "开始配置检索平台模块"
./module/change_search_ip.sh ${server_ip}

printGreenText "开始配置服务管理模块"
./module/change_hikinfer_ip.sh ${server_ip}

printGreenText "Change Success， 稍等片刻，等待服务启动成功"


