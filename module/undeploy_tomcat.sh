#!/bin/bash
# author：songquanheng
# 2020-12-23 11:14:27
# 脚本用途： 该脚本用于部署tomcat
# 1、删除/opt/tomcat/目录


cd $(dirname $0)

. ../util/util_function.sh
printGreenText "导入功能函数，部署tomcat"
. ../conf/camp.conf
printGreenText "卸载tomcat运行环境,请稍等..."
sleep 1
if [ -d ${tomcat_path}/tomcat/ ]; then
	rm -rf ${tomcat_path}/tomcat/
fi

echo
echo
