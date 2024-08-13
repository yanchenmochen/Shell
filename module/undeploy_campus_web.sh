#!/bin/bash
# author：songquanheng
# 2020年12月23日11:50:24
# 脚本用途： 该脚本用于卸载AI推理平台前端程序
# 1、从camp.conf获取web程序安装路径
# 2、以安装顺序相反的顺序移除
# 2.1、 卸载推流程序
# 2.2、 卸载运行管理中心程序
# 2.3、 卸载AI推理主页面程序
# 
cd $(dirname $0)
. ../util/util_function.sh
printGreenText "导入功能函数，部署tomcat"
. ../conf/camp.conf
echo "zysbfx_web_path=${zysbfx_web_path}"

printGreenText "导入功能函数，准备卸载页面程序"
printGreenText "卸载过程较为耗时，请耐心等待..."

printGreenText "卸载hjj适配物联网任务创建页面"
netstat -anop | grep 7909 > /dev/null
if [ $? -eq 0 ]; then
	bash ./portkill 7909
	
fi

sleep 1
rm -rf ${zysbfx_web_path}/campus-hjj-task


# 导入配置文件，获取安装路径
printGreenText "卸载推流程序"

sleep 1

systemctl disable campus-preview
systemctl stop campus-preview

netstat -anop | grep -w 8888 > /dev/null
if [ $? -eq 0 ]; then
	/usr/bin/pkill -f campus-preview
	
fi
rm -rf ${zysbfx_web_path}/rtsp_server
rm -rf /etc/systemd/system/campus-preview.service

rtsp_server_config=/etc/ld.so.conf.d/ffmpeg.conf
if [ -e ${rtsp_server_config} ]; then
	rm -rf ${rtsp_server_config}
	/sbin/ldconfig
fi
printGreenText "移除推流程序所需要的ffmepg，较为耗时，请等待..."
sleep 2
rm -rf ${zysbfx_web_path}/ffmpeg



printGreenText "卸载运行管理中心页面程序..."
sleep 1
systemctl disable campus-front
systemctl stop campus-front

netstat -anop | grep -w 8091 > /dev/null
if [ $? -eq 0 ]; then
	bash ./portkill 8091
fi
rm -rf ${zysbfx_web_path}/manage_center

printGreenText "卸载AI推理主页面程序"
netstat -anop | grep -w 8090 > /dev/null
if [ $? -eq 0 ]; then
	bash ./portkill 8090
fi
rm -rf ${zysbfx_web_path}/campus-web
rm -rf /etc/systemd/system/campus-front.service
sleep 1
rm -rf ${zysbfx_web_path}

echo
echo
