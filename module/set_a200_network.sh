#!/bin/bash
#
#该脚本用于设备发现模块修改A200的ip、掩码、网关、hostname等信息
#

if [ "$1" = "-help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "--help" ] ; then
        STATUS_MODE="HELP"
elif [ "$1" = "help" ] ; then
        STATUS_MODE="HELP"
fi

cd $(dirname $0)
pwd
. ../util/util_function.sh
. ../conf/camp.conf
desc() {
    printRedText "脚本的主要作用包括："
	printGreenText "修改IP：-i 10.0.96.186 修改子网掩码：-n xxxx  修改网关：-g zzzz 修改主机名：-h aaaa  可单独使用一个，也可全部使用"
}

if [ "$STATUS_MODE" = "HELP" ] ; then
    desc
	echo "脚本用法： bash set.sh -i 10.0.96.186 -n xxxx -g zzzz -h aaaa"
    exit 1
fi


while getopts ':i:n:g:h:' opt
do
  case $opt in
        i)
        echo "输入的IP: $OPTARG"
		sed -i '/address /c address '"$OPTARG" /etc/network/interfaces
        ;;
		
        n)
        echo "输入的netmask: $OPTARG"
		sed -i '/netmask /c netmask '"$OPTARG" /etc/network/interfaces
        ;;
		
        g)
		echo "输入的gateway: $OPTARG"
        sed -i '/gateway /c gateway '"$OPTARG" /etc/network/interfaces
        ;;
		
		h)
		echo "输入的hostname: $OPTARG"
        echo "$OPTARG" > /etc/hostname
        ;;
		
        *)
        exit 1;;
    esac
done