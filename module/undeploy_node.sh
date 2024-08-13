#!/bin/bash
# author：songquanheng
# 2020-12-24 09:23:24
# 脚本用途： 该脚本用于卸载node程序
# 1、从camp.conf获取node安装路径
# 3、移除环境变量
# 4、移除node程序包


cd $(dirname $0)
. ../util/util_function.sh
printGreenText "导入功能函数，部署tomcat"
. ../conf/camp.conf

echo "node_path=${node_path}"
echo "config_file_path=${config_file_path}"

if grep "NODE_HOME" ${config_file_path}; then
	printGreenText "移除node环境变量"
	sed -i '/NODE_HOME/d' ${config_file_path}
fi

cat ${config_file_path} | grep "NODE_HOME"
printGreenText "移除node程序包较为耗时，请等待..."
sleep 2
rm -rf ${node_path}/node
