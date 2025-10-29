#!/bin/bash
#owner: zhangpengwei
#date: 2020年11月30日10:41:07
#脚本作用：该脚本用于在服务器上部署expect环境，将tcl的安装包和expect的安装包、安装脚本拷贝到/root目录下
# 

cd $(dirname $0)
pwd

# 将expect目录下安装包和安装脚本复制到root目录下
expect_path='./package/expect/*'
install_expect_path='./module/install_expect.sh'
cp ${expect_path} /root
cp ${install_expect_path} /root


# 进入root目录,赋予脚本执行权限
cd /root
chmod 777 install_expect.sh

# 调用install_expect.sh脚本，安装expect程序
bash install_expect.bash





