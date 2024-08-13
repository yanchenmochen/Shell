#!/bin/bash
# author：zhangpengwei
# 2020-11-24 10:04:39
# 脚本用途： 该脚本用于部署卸载nginx程序
# 1. 卸载/usr/local/nginx目录
# 2. 卸载/opt/nginx目录

cd $(dirname $0)
. ../util/util_function.sh
printGreenText "导入功能函数，卸载nginx"
. ../conf/camp.conf
echo "web_program_path=${web_program_path}"

printGreenText "导入功能函数，准备nginx源码编译目录"
rm -rf /usr/local/nginx/
printGreenText "卸载nginx源码目录"
rm -rf ${web_program_path}/nginx
printGreenText "完成nginx的卸载"
echo
echo
