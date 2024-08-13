#!/bin/bash
#owner: songquanheng
#date: 2020年11月17日14:09:27
#脚本作用：该脚本用于一键在A200芯片上安装torque程序以及jlib.tar中依赖
#注意： 该脚本位于shell文件中。
#  1. 安装torque
#  2. 解压jlib.tar到目录中


if [ "$1" = "-help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "--help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "help" ] ; then
        STATUS_MODE="HELP"
fi

if [ "$STATUS_MODE" = "HELP" ] ; then
    echo "该脚本用于一键在A200芯片上安装torque程序以及jlib.tar中依赖"
    exit 1
fi

cd $(dirname $0)
pwd
. ../util/util_function.sh
printGreenText "导入功能函数"

. ../conf/camp.conf
echo "config_file_path=${config_file_path}"

# 判断当前用户是否为root
printYellowText "当前用户${USER}"

if test ${USER} != "root" ;then
  printGreenText "请使用root用户执行ip配置操作"
  exit 4
fi

torque_clients=../package/torque-cluster1-clients-linux-aarch64.sh
torque_devel=../package/torque-cluster1-devel-linux-aarch64.sh
torque_mom=../package/torque-cluster1-mom-linux-aarch64.sh
torque_server=../package/torque-cluster1-server-linux-aarch64.sh

if [ ! -e ${torque_clients} ]; then
    printYellowText "torque-clients不存在"
    exit 5
fi

if [ ! -e ${torque_devel} ]; then
    printYellowText "${torque_devel}不存在"
    exit 5
fi

if [ ! -e ${torque_mom} ]; then
    printYellowText "${torque_mom}不存在"
    exit 5
fi

if [ ! -e ${torque_server} ]; then
    printYellowText "${torque_server}不存在"
    exit 5
fi

# 为安装脚本赋予可执行权限
chmod +x ${torque_clients}
chmod +x ${torque_devel}
chmod +x ${torque_mom}
chmod +x ${torque_server}

# 执行安装程序
sh ${torque_server} --install
sh ${torque_devel} --install
sh ${torque_clients} --install
sh ${torque_mom} --install

lib_dir=/home/jdp/atlas/toruqe-arm/lib/
jlib_resource_path=../package/jlib.tar

if [ ! -e ${lib_dir} ] ;then
	printRedText "torque程序安装失败"
	exit 5
fi

tar xfv ${jlib_resource_path} -C ${lib_dir}
printGreenText "jlib包中依赖均导入到${lib_dir}中"
printGreenText "${lib_dir}目录中共有文件: `ls -al | wc -l`个"

printGreenText "为芯片$(ip)配置toruqe-arm环境变量"
if grep "toruqe-arm" ${config_file_path} > /dev/null ; then
	printGreenText "在配置文件中已经存在了环境变量"
else
	# 在配置文件中找不到toruqe-arm
	printGreenText "# 在配置文件中找不到toruqe-arm环境变量"
	# 添加环境变量
	echo >> ${config_file_path}
	echo 'export PATH=/home/jdp/atlas/toruqe-arm/sbin:$PATH' >> ${config_file_path}
	echo 'export PATH=/home/jdp/atlas/toruqe-arm/bin:$PATH' >> ${config_file_path}
	echo 'export LD_LIBRARY_PATH=/home/jdp/atlas/toruqe-arm/lib:$LD_LIBRARY_PATH' >> ${config_file_path}
	echo 'export CPATH=/home/jdp/atlas/toruqe-arm/include:$CPATH' >> ${config_file_path}
fi

printGreenText "成功安装toruqe程序，并且导入jlib依赖，并完成对环境变量的配置"







