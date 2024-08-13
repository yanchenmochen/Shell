#/bin/sh

# date: 2020年4月20日18:30:17
# author: songquanheng
# desc: 该脚本用于一键部署expect环境，需要tcl的安装和expect的安装包。
# 注意：把安装包tcl8.7a1-src.tar.gz和expect-5.45.tar.gz与当前脚本均放置在/root目录下

install_home=/root
tar xf tcl8.7a1-src.tar.gz
cd /root/tcl8.7a1/unix
./configure --prefix=/usr/tcl --enable-shared
make && make install
echo "tcl 安装成功"
# Installing library files to /usr/tcl/lib/tcl8.7/
# Installing package http1.0 files to /usr/tcl/lib/tcl8.7/http1.0/
# Installing package http 2.8.12 as a Tcl Module
# Installing package opt0.4 files to /usr/tcl/lib/tcl8.7/opt0.4/
# Installing package msgcat 1.6.1 as a Tcl Module
# Installing package tcltest 2.4.0 as a Tcl Module
# Installing package platform 1.0.14 as a Tcl tcModule
# Installing package platform::shell 1.1.4 as a Tcl Module
# Installing encoding files to /usr/tcl/lib/tcl8.7/encoding/
# Making directory /usr/tcl/lib/tcl8.7/msgs
# Installing message catalog files to /usr/tcl/lib/tcl8.7/msgs/
# Making directory /usr/tcl/share/man
# Making directory /usr/tcl/share/man/man1
# Making directory /usr/tcl/share/man/man3
# Making directory /usr/tcl/share/man/mann
# Installing and cross-linking top-level (.1) docs to /usr/tcl/share/man/man1/
# Installing and cross-linking C API (.3) docs to /usr/tcl/share/man/man3/
# Installing and cross-linking command (.n) docs to /usr/tcl/share/man/mann/
# Making directory /usr/tcl/include
# Installing header files to /usr/tcl/include/

# 安装完毕以后进入tcl源代码根目录，把子目录unix下面的tclUnixPort.h拷贝到子目录generic中，expect的安装过程还需要用
cp tclUnixPort.h ../generic/
cd ${install_home}
tar xf expect-5.45.tar.gz
# 进入expect根目录
# 注意：在飞腾服务器上运行时可能会弹出错误：configure: error cannot guess build type;you must specify one，
# 此时需要指定--build=arm-linux
cd expect-5.45/
./configure --prefix=/usr/expect --with-tcl=/usr/tcl/lib --with-tclinclude=../tcl8.7a1/generic/ --build=arm-linux

# 显示如下的内容表示编译成功
# checking dlfcn.h presence... yes
# checking for dlfcn.h... yes
# checking sys/param.h usability... yesex
# checking sys/param.h presence... yes
# checking for sys/param.h... yes
# configure: creating ./config.status
# config.status: creating Makefile

make && make install
# 安装完成之后做一个软连接
ln -s /usr/tcl/bin/expect /usr/expect/bin/expect
# 加入环境变量并测试
echo 'export PATH=$PATH:/usr/expect/bin' >> /etc/profile

# 重建一个会话，让环境变量的配置生效
# root@cetc52-SYS-4029GP-TRT2:~# expect
# expect1.1> 