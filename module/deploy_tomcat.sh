#!/bin/bash
# author：zhangpengwei
# 2020-11-24 10:04:39
# 脚本用途： 该脚本用于部署tomcat
# 1、从camp.conf获取tomcat安装路径
# 2、创建对应目录
# 3、解压tomcat压缩包 

cd $(dirname $0)

. ../util/util_function.sh
printGreenText "导入功能函数，部署tomcat"
. ../conf/camp.conf
resource_path='../package/apache-tomcat-8.5.43.tar.gz'

echo "tomcat_path=${tomcat_path}"

if [ ! -e ${tomcat_path} ]; then
    printYellowText "tomcat待部署的路径为: ${tomcat_path}"
    mkdir -p ${tomcat_path}
fi

printYellowText "解压文件: " ${resource_path}
printGreenText "解压文件较为耗时，请等待..."
sleep 3
tar xf ${resource_path} -C ${tomcat_path}
printYellowText "把解压过的文件夹重命名为${tomcat_path}/tomcat"

# 将apache-tomcat-8.5.43重命名为tomcat
# 注意如果目录下没有tomcat文件夹将会重命名，否则会将文件夹移动到tomcat目录下
mv ${tomcat_path}/apache-tomcat-8.5.43 ${tomcat_path}/tomcat
printYellowText "Tomcat部署完成，当前Tomcat根目录位于: ${tomcat_path}/tomcat" 
echo
echo
