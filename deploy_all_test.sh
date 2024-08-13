#!/bin/bash
#owner: zhangpengwei
#date: 2020年11月25日15:41:07
#脚本作用：该脚本用于在服务器上一键部署资源智能识别程序
#  调用module文件夹中脚本依次部署
#  tomcat、后端war包程序、node、nginx、前端程序、ES



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
. ./util/util_function.sh
printGreenText "导入功能函数"

# 判断当前用户是否为root
printYellowText "当前用户${USER}"

if test ${USER} != "root" ;then
  printGreenText "请使用root用户执行ip配置操作"
  exit 4
fi

# 调用deploy_tomcat.sh脚本，部署tomcat程序
sh ./module/deploy_tomcat.sh

# 调用deploy_campus.sh脚本，部署campus程序
# sh ./module/deploy_campus.sh

# 调用deploy_node.sh脚本，部署node
# sh ./module/deploy_node.sh

# 调用deploy_nginx.sh脚本，部署nginx
# sh ./module/deploy_nginx.sh

# 调用deploy_campus_web.sh脚本，部署web程序
# sh ./module/deploy_campus_web.sh

# 调用deploy_es.sh脚本，部署es
# sh ./module/deploy_es.sh





