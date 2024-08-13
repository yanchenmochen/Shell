#!/bin/bash

# owner: songquanheng
# date: 2021年7月5日15:20:06
# 脚本作用： 该脚本用于为管理节点添加符号链接

cd $(dirname $0)
module_path=$(pwd)

. ../util/util_function.sh
printGreenText "为系统添加快捷命令find_soft_link_real_path"
ln -s ${module_path}/find_soft_link_real_path.sh /usr/sbin/find_soft_link_real_path
printGreenText "为系统建立快捷命令campus-version"
ln -s ${module_path}/campus-version.sh /usr/sbin/campus-version
printGreenText "为系统添加快捷命令campstat"
ln -s ${module_path}/campstat /usr/sbin/campstat
printGreenText "为系统添加快捷命令campus-log"
ln -s ${module_path}/campus-log /usr/sbin/campus-log

printGreenText "为系统添加快捷命令campus-config"
ln -s ${module_path}/campus-config.sh /usr/sbin/campus-config

printGreenText "为系统添加快捷命令campus-update"
ln -s ${module_path}/campus-update.sh /usr/sbin/campus-update

printGreenText "为系统添加快捷命令campus-err"
ln -s ${module_path}/campus-err.sh /usr/sbin/campus-err

printGreenText "为系统添加快捷命令campus-shell"
ln -s ${module_path}/campus-shell.sh /usr/sbin/campus-shell
