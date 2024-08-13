#!/bin/bash

# owner: songquanheng
# date: 2021年7月5日15:20:06
# 脚本作用： 该脚本用于为卸载系统安装的符号链接

cd $(dirname $0)
current_dir=$(pwd)

. ../util/util_function.sh

printGreenText "删除为系统添加的快捷命令campus-shell"
rm -rf /usr/sbin/campus-shell

printGreenText "删除为系统添加的快捷命令campus-err"
rm -rf /usr/sbin/campus-err

printGreenText "删除为系统添加的快捷命令campus-log"
rm -rf /usr/sbin/campus-log

printGreenText "删除为系统添加的快捷命令campstat"
rm -rf /usr/sbin/campstat

printGreenText "删除为系统添加的快捷命令campus-config"
rm -rf /usr/sbin/campus-config

printGreenText "删除为系统添加的快捷命令campus-update"
rm -rf /usr/sbin/campus-update
printGreenText "删除为系统添加的快捷命令campus-version"
rm -rf /usr/sbin/campus-version
printGreenText "删除为系统添加的快捷命令find_soft_link_real_path"
rm -rf /usr/sbin/find_soft_link_real_path
