#!/bin/bash
# author：zhangpengwei
# 2020-11-24 10:04:39
# 脚本用途： 该脚本用于部署资源智能识别后端程序
# 1、从camp.conf获取后端程序安装路径
# 2、创建对应目录
# 3、将程序包放到指定位置
# campus程序包：ROOT.war; 
# ISC取流程序包：platform.war;
# 注意： 该脚本仅复制文件，不启动后台程序

cd $(dirname $0)

. ../util/util_function.sh
printGreenText "导入功能函数，部署campus"

# 导入配置文件，获取安装路径
. ../conf/camp.conf 
# camp_path变量保存到是war包所在目录
echo "camp_path=${camp_path}"

if [ ! -e ${camp_path} ]; then
    printYellowText "未找到tomcat，请先安装tomcat"
	printRedText "请先安装Tomcat环境，执行$(pwd)/deploy_tomcat.sh"
    exit 5
fi

sleep 2
# 删除webapps目录下的ROOT文件夹和platform文件夹
rm -rf ${camp_path}/ROOT
# rm -rf ${camp_path}/platform


# 将war包放到webapps目录下
camp_war='../package/ROOT.war'
platform_war='../package/platform.war'
cp ${camp_war} ${camp_path}
printGreenText "主程序安装完成"
#cp ${platform_war} ${camp_path}
echo
echo
