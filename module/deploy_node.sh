#!/bin/bash
# author：zhangpengwei
# 2020-11-24 10:04:39
# 脚本用途： 该脚本用于部署node程序
# 1、从camp.conf获取node安装路径
# 2、创建对应目录
# 3、解压node压缩包 
# 4、配置环境变量

cd $(dirname $0)


. ../util/util_function.sh
printGreenText "导入功能函数，部署node"
. ../conf/camp.conf 
resource_path='../package/node-v10.22.1-linux-arm64.tar.gz'


echo "node_path=${node_path}"
echo "config_file_path=${config_file_path}"

if [ ! -e ${node_path} ]; then
    printYellowText "node待部署的路径为: ${node_path}"
    mkdir -p ${node_path}
fi

printYellowText "安装实时预览运行环境: " ${resource_path}
tar xf ${resource_path} -C ${node_path}

printYellowText "重命名为node"
cd ${node_path}
mv node-v10.22.1-linux-arm64 node

printYellowText "配置环境变量"

# 在配置文件中找不到NODE_HOME
if grep -q -v "NODE_HOME" ${config_file_path}; then
	printGreenText "在配置文件中添加NODE环境变量"
	echo >> ${config_file_path}
	echo 'export NODE_HOME=/opt/node'>>${config_file_path}
	echo 'export PATH=$PATH:${NODE_HOME}/bin'>>${config_file_path}
else
	printGreenText "在配置文件中已经存在了环境变量"
fi
cat ${config_file_path} | grep "NODE_HOME"

. ${config_file_path}

node -v
if [ $? -eq 0 ]; then 
	printRedText "node安装成功"
else
	printRedText "node安装失败，请检查"
fi

