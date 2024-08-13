#!/bin/bash
# author：songquanheng
# 2021年7月16日10:46:21
# 脚本用途： 该脚本用于切换到脚本顶层目录
# 使用bash调用该脚本无效，相当于调用这个脚本的时候，系统会开启另外一个终端，然后执行完后，
# 再次回到之前的终端。
# 该脚本的方式为source campus-shell或者. campus-shell

# 判断是否经由软连接命令执行
real_work_dir=`find_soft_link_real_path campus-shell`

if [[ ${real_work_dir} != "" && ${real_work_dir} != `pwd` ]];then
    cd $(dirname "${real_work_dir}")
fi
