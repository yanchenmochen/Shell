#!/bin/bash
#Author: XuYumin
#Date: 2021/08/25
#Description: auto make bond0 script


net_list=$(ifconfig -a | grep -i enp | awk '{print $1}' | sed 's/://g')

cat << INFO
=========自动配置双网口绑定=========
本机检测到以下网口：
INFO
j=0
for net in $net_list
do
    eth_list[j]=$net
    if [ -z "`ethtool $net | grep "10000"`" ];then
        echo "[千]${j}.${net}"
    else
        echo "[万]${j}.${net}"
    fi
    ((j++))
done
echo "=============================="
while true
do
read -p "请选择第一个绑定网口序号：" eth1
if [ $eth1 -gt 0 -a $eth1 -lt $j ];then
    break
else
    echo "输入序号超出范围，请重新输入！"
fi
done
while true
do
read -p "请选择第二个绑定网口序号：" eth2
if [ $eth2 -gt 0 -a $eth2 -lt $j ];then
    if [ $eth2 -ne $eth1 ];then
        break
    else
        echo "第一个绑定网口序号和第二个绑定网口序号相同，请重新输入！"
    fi
else
    echo "输入序号超出范围，请重新输入！"
fi
done
read -p "请输入绑定网口名称(如bond0)：" eth_name
read -p "请输入绑定网口ip地址：" IP
cat << INFO
请输入bonding策略：
1	主备模式
6	负载均衡模式
INFO
read -p "请输入bonding策略序号：" cata
echo "即将生成绑定网口"

cat << INFO >> /etc/network/interfaces
auto ${eth_list[$eth1]}
iface ${eth_list[$eth1]} inet manual
bond-master $eth_name

auto ${eth_list[$eth2]}
iface ${eth_list[$eth2]} inet manual
bond-master $eth_name

auto $eth_name
iface $eth_name inet static
address $IP
netmask 255.255.255.0
bond-slaves none
bond-mode $cata
bond-miimon 100
INFO

if [ -z "`grep bonding /etc/modules`" ];then
    echo bonding >> /etc/modules
fi
echo "绑定网口已完成配置，重启后完全生效"

