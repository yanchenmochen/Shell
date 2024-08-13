#!/bin/bash
# author：zhangpengwei
# 2020-11-24 10:04:39
# 脚本用途： 该脚本用于部署资源智能识别前端web程序
# 1、从camp.conf获取web程序安装路径
# 2、创建对应目录
# 3、解压web程序压缩包 
# campus前端程序：dist.zip;  运管中心：manage_center.zip; rtsp推流：rtsp_server.zip
# 4、启动nginx程序
# 5、启动node程序

cd $(dirname $0)

. ../util/util_function.sh
printGreenText "导入功能函数，部署web程序"
printGreenText "部署的web程序包括AI推理主页面，运管中心页面和实时预览程序"

# 导入配置文件，获取安装路径
. ../conf/camp.conf 
echo "zysbfx_web_path=${zysbfx_web_path}"

if [ ! -e ${zysbfx_web_path} ]; then
    printYellowText "web程序安装的路径为: ${zysbfx_web_path}"
    mkdir -p ${zysbfx_web_path}
	mkdir -p ${zysbfx_web_path}
fi

# 解压campus前端程序到指定位置
dist_path='../package/campus-web.zip'
printGreenText "安装AI推理主页面程序: " ${dist_path}
sleep 2
unzip -qO UTF-8 ${dist_path} -d ${zysbfx_web_path}
campus_web_service='../service/campus-front.service'
cp ${campus_web_service} /etc/systemd/system/

# 解压运管中心前端程序到指定位置
manage_center_path='../package/manage_center.zip'
sleep 2
printGreenText "安装运管中心前端程序: " ${manage_center_path}
unzip -q ${manage_center_path} -d ${zysbfx_web_path}

# 解压rtsp推流前端程序到指定位置
rtsp_server_path='../package/rtsp_server.zip'
printGreenText "安装实时预览程序: " ${rtsp_server_path}

sleep 2
unzip -q ${rtsp_server_path} -d ${zysbfx_web_path}
# 修改实时推流程序server.js写死的路径
sed -i "s|/zysbfx/ffmpeg/bin/ffmpeg|${zysbfx_web_path}/ffmpeg/bin/ffmpeg|g" ${zysbfx_web_path}/rtsp_server/server.js

rtsp_server_service='../service/campus-preview.service'
cp ${rtsp_server_service} /etc/systemd/system/

rtsp_server_config=/etc/ld.so.conf.d/ffmpeg.conf
if [ ! -e ${rtsp_server_config} ]; then
	touch ${rtsp_server_config}
	echo "${zysbfx_web_path}/ffmpeg/lib" > ${rtsp_server_config}
	/sbin/ldconfig
fi
chmod +x ${zysbfx_web_path}/ffmpeg/bin/ffmpeg
chmod +x ${zysbfx_web_path}/ffmpeg/bin/ffprobe

printGreenText "安装AI推理平台 hjj适配物联网平台任务管理页面"
hjj_task_path='../package/campus-hjj-task.zip'
unzip -qO UTF-8 ${hjj_task_path} -d ${zysbfx_web_path}

systemctl daemon-reload
printGreenText "AI推理服务前端推理程序安装完成"