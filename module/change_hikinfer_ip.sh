#!/bin/bash
#owner: songquanheng
#date: 2020年5月14日 14点51分于阿拉善百吉宾馆
#脚本作用： 该脚本用于修改HikInfer启动时需要的配置文件。
#基本流程是先删除配置文件，然后重新创建

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

if [ "$STATUS_MODE" = "HELP" ] ; then
    printRedText "脚本运行时需要添加1个参数：一体机ip"
	echo "./change_hikinfer_ip server_ip"
	exit 1
fi

if [ $# -ne 1 ]; then
    printRedText "脚本运行时需要添加1个参数：一体机ip"
	echo "./change_hikinfer_ip server_ip"
    exit 5
fi

echo "进行参数的校验"
if test $(valid_ip "$1") != $TRUE ;then 
    printRedText "请输入合法而且真实一体机ip地址"
    printYellowText "please input -help option to get some help"
    exit 5
fi

#开始推理服务修改IP的功能
config=/hikmars3/system/logs/config.json
rm -rf $config
touch $config
chmod 777 $config
echo "{\"server_ip\":\"$1\"}" >> $config
printGreenText "重启k8s服务"
#完成推理服务修改IP的功能
cd /home/cetc52/k8sflask/k8s_flask_245
# 请确保当前五推理服务和任务正在运行。
echo "重启推理服务管理模块"
kubectl delete -f deploy.yaml
kubectl create -f deploy.yaml
echo "重启推理服务管理模块"
