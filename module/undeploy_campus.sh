#!/bin/bash
# author：songquanheng
# 2020-12-23 10:56:31
# 脚本用途： 该脚本用于停止tomcat服务和其中的程序
# 1、停止tomcat程序
# 2、删除相应的文件夹/opt/tomcat

cd $(dirname $0)

. ../util/util_function.sh
printGreenText "导入功能函数，部署campus"

# 导入配置文件，获取安装路径
. ../conf/camp.conf 
# camp_path变量保存到是war包所在目录
echo "tomcat_path=${tomcat_path}"
printGreenText "停止主程序的运行"
netstat -anop | grep 8080
if [ $? -eq 0 ] ;then
	printYellowText "主程序正在运行"
	bash ${tomcat_path}/tomcat/bin/shutdown.sh
	./portkill 8080
else
	printGreenText "主程序并未运行"
fi


printGreenText "卸载相关程序，请稍等"
sleep 2
rm -rf ${tomcat_path}/tomcat/webapps/ROOT*
echo
echo
