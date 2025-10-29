#!/bin/bash
# author: songquanheng
# date: 2020-04-22 
# desc: 该脚本用于推送root公钥到指定的ip列表
#    注意： expect和tcl需要提前安装	

if [ $# -eq 0 ] || [ $1 == "-h" ] || [ $1 == "--help" ] ;then
	echo "请键入待推送公钥的ip列表"
	echo "用法示例: (./push_pub.sh 9.9.9.39 9.9.9.35)"
	exit 13 
fi

if [ ! -e ~/.ssh/id_rsa ]; then
	ssh-keygen -P "" -f ~/.ssh/id_rsa
fi

echo $#
echo $*
for  i in $@ 
do
	echo $i
	expect <<-EOF
	spawn ssh-copy-id root@$i
	expect {
		"yes/no" {send "yes\r"; exp_continue}
		"password" {send "123123\r"}
	}
	send "ifconfig | grep 10.0.96\r"
	expect "~ #" send "exit"
	EOF
done
echo "成功把公钥推送到了如下的ip"

for i in $@
do
	echo $i
done
