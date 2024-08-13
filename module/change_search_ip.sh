#!/bin/bash
# 本脚本的运行环境为aarch64架构的Ubuntu（kylin）系统,实现检索平台服务修改IP的功能
#运行该脚本的前提：elasticsearch数据库已安装、检索平台服务已安装
# 参数如下：
#$1:新IP


if [ "$1" = "-help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "--help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "help" ] ; then
        STATUS_MODE="HELP"
fi


if [ "$STATUS_MODE" = "HELP" ] ; then
    echo "脚本运行时需要添加1个参数：一体机ip"
	echo "./change_search_ip server_ip"
	exit 1
fi
cd $(dirname $0)
pwd
. ../util/util_function.sh
if [ $# -ne 1 ]; then
    printRedText "脚本运行时需要添加1个参数：一体机ip"
	echo "./change_search_ip server_ip"
    exit 5
fi

echo "进行参数的校验"
if test $(valid_ip "$1") != $TRUE ;then 
    printRedText "请输入合法而且真实一体机ip地址"
    printYellowText "please input -help option to get some help"
    exit 5
fi

ip=$1

#停止Elasticsearch服务和检索平台服务
echo "停止Elasticsearch服务和检索平台服务..."
systemctl stop searchx.service
systemctl stop elasticsearch.service

#修改配置
echo "修改配置文件..."
sed -i '/^es.network/d' /etc/usearch/search.yml
echo "es.network: ${ip}" >> /etc/usearch/search.yml


sed -i '/^network.host/d' /etc/elasticsearch/elasticsearch.yml
echo "network.host: ${ip}" >> /etc/elasticsearch/elasticsearch.yml

#重新启动Elasticsearch服务和检索平台服务
echo "重新启动Elasticsearch服务和检索平台服务..."
systemctl start elasticsearch.service
systemctl start searchx.service
echo "检索平台IP修改完成！"