#!/bin/bash

# 判断是否经由软连接命令执行
real_work_dir=`find_soft_link_real_path campus-version`

if [[ ${real_work_dir} != "" && ${real_work_dir} != `pwd` ]];then
    cd ${real_work_dir}
fi


. ../conf/camp.conf
. ../util/util_function.sh
printGreenText "version: $version"
