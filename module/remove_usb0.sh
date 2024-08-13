#!/bin/bash

# owner: songquanheng
# date: 2020年11月17日15:03:59于文福路9号
# 警告： 需要已经处理过每个板子上多余的usb0网卡信息。
# 脚本作用： 该脚本会移除/etc/network/interfaces从第十一到第十五行属于usb0虚拟网卡的信息。

cat /etc/network/interfaces
line_no=$(grep -n 'auto usb0' /etc/network/interfaces)
echo "auto usb0位于${line_no}行"
if grep 'auto usb0' /etc/network/interfaces > /dev/null; then
	sed -i "11,15d" /etc/network/interfaces
fi
echo 

echo "删除auto usb0虚拟网卡信息之后"
cat /etc/network/interfaces

