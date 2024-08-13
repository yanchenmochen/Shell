#!/bin/bash

#owner: yangbang
#date: 2021年7月15日19:28:32
#脚本作用：该脚本用于获取符号链接命令的真实位置的上一层目录。

# 获取符号链接命令的真实位置
# $1 为创建的符号链接 例如campus-log.
# 注意： 所有的符号链接命令均放置在/usr/sbin目录下。这是先验知识。
# 函数会返回符号链接命令的真实位置的父目录 比如说campus-log符号链接了/home/sqh/shell/module/campus-log
# 则函数会返回/home/sqh/shell/module/
shell_file_real_path=`readlink -f "/usr/sbin/$1"`
echo $(dirname "${shell_file_real_path}")