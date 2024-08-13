#!/bin/bash
# author：zhangpengwei
# 2020-11-24 10:04:39
# 脚本用途： 该脚本用于部署nginx程序
# 1、从camp.conf获取nginx安装路径
# 2、创建对应目录
# 3、解压nginx压缩包 
# 4、源码编译安装nginx

cd $(dirname $0)
current_dir=$(pwd)

. ../util/util_function.sh
printGreenText "导入功能函数，部署nginx"

. ../conf/camp.conf
sleep 1 
printGreenText "解压Nginx源码包，较为耗时，请等待"
resource_path='../package/nginx/nginx-1.22.0.tar.gz'

# 页面程序
front_path=${web_program_path}

echo "front_path=${front_path}"

echo "zysbfx_web_path=${zysbfx_web_path}"

if [ ! -e ${front_path} ]; then
    printYellowText "nginx待部署的路径为: ${front_path}"
    mkdir -p ${front_path}
fi

printYellowText "解压文件: " ${resource_path}
tar xf ${resource_path} -C ${front_path}

printYellowText "重命名为${front_path}/nginx"
mv ${front_path}/nginx-1.22.0 ${front_path}/nginx

printGreenText "需要编译nginx运行环境，使用默认环境。较为耗时，请等待..."
sleep 2
cd ${front_path}/nginx
printGreenText "执行configure命令"
./configure > /dev/null

printGreenText "执行make命令"
cd ${front_path}/nginx
make > /dev/null
printGreenText "执行make install命令"
cd ${front_path}/nginx
make install > /dev/null
printGreenText "nginx运行环境已经安装成功"
printGreenText "拷贝并修改nginx配置文件"

# 切换到当前目录，才能使用相对路径
cd ${current_dir}
nginx_source_config_file='../package/nginx/nginx.conf'
nginx_dest_config_file=/usr/local/nginx/conf/nginx.conf
cp ${nginx_source_config_file} /usr/local/nginx/conf/
sed -i "s|zysbfx_web_path|${zysbfx_web_path}|g" ${nginx_dest_config_file}
cat ${nginx_dest_config_file} | grep -E "listen|root"
printGreenText "nginx安装根目录位于: /usr/local/nginx/"