#!/bin/bash
# author: songquanheng
# date: 2020-04-22 
# desc: �ýű���������root��Կ��ָ����ip�б�
#    ע�⣺ expect��tcl��Ҫ��ǰ��װ	

if [ $# -eq 0 ] || [ $1 == "-h" ] || [ $1 == "--help" ] ;then
	echo "���������͹�Կ��ip�б�"
	echo "�÷�ʾ��: (./push_pub.sh 9.9.9.39 9.9.9.35)"
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
echo "�ɹ��ѹ�Կ���͵������µ�ip"

for i in $@
do
	echo $i
done
