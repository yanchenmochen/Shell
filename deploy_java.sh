#!/bin/bash
#owner: songquanheng
#date: 2020年11月12日17:51:07
#脚本作用：该脚本用于在每个A200上一键部署jdk环境
#  1. 创建目录 /usr/local/java
#  2. 解压jdk-8u261-linux-arm64-vfp-hflt.tar.gz到java目录下
#  3. 配置环境变量/etc/profile
#  4. 虽然在脚本上导入了/etc/profile，但由于控制台是一个shell，脚本是它的子shell。两者不是一个环境。
#     因此需要单独在启一个shell，使得配置文件生效。


if [ "$1" = "-help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "--help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "help" ] ; then
        STATUS_MODE="HELP"
fi

if [ "$STATUS_MODE" = "HELP" ] ; then
    echo "请首先修改conf/camp.conf配置文件,获得java要安装的目录"
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


. ./conf/camp.conf 
resource_path='./package/jdk-8u261-linux-arm64-vfp-hflt.tar.gz'

echo "java_path=${java_path}"
echo "config_file_path=${config_file_path}"

if [ ! -e ${java_path} ]; then
    printYellowText "java待部署的路径为: ${java_path}"
    mkdir -p ${java_path}
fi

printYellowText "解压文件: " ${resource_path}
tar xvf ${resource_path} -C ${java_path}
printYellowText "配置环境变量"


# 在配置文件中找不到JAVA_HOME
if grep "JAVA_HOME" ${config_file_path}; then
	printGreenText "在配置文件中已经存在了环境变量"
else
    printGreenText "在配置文件中添加JAVA环境变量"
	echo 'export JAVA_HOME=/usr/local/java/jdk1.8.0_261'>>${config_file_path}
	echo 'export JAVA_HOME=/usr/local/java/jdk1.8.0_261'>>${config_file_path}
	echo 'export JRE_HOME=${JAVA_HOME}/jre'>>${config_file_path}
	echo 'export CLASSPATH=.:${JAVA_HOME}/lib:{JRE_HOME}/lib'>>${config_file_path}
	echo 'export PATH=${JAVA_HOME}/bin:$PATH'>>${config_file_path}
fi

. ${config_file_path}

java -version
if [ $? -eq 0 ]; then 
	printRedText "Java 8安装成功"
else
	printRedText "Java 8安装失败，请检查"
fi
 

