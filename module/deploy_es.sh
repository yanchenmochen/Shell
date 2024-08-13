#!/bin/bash
# author：zhangpengwei
# 2020-11-24 10:04:39
# 脚本用途： 该脚本用于部署ES程序
# 1、从camp.conf获取ES安装路径
# 2、创建对应目录
# 3、解压ES压缩包 
# 4、将searchx的配置文件放到指定位置
# 5、将service文件放到指定位置

cd $(dirname $0)

. ../util/util_function.sh
printGreenText "导入功能函数，部署ES"
. ../conf/camp.conf

resource_path='../package/deploy_aarch64.zip'

echo "es_path=${es_path}"

if [ ! -e ${es_path} ]; then
    printYellowText "ES安装的路径为: ${es_path}"
    mkdir -p ${es_path}
fi

printYellowText "安装elasticsearch和search平台: " ${resource_path}
printGreenText "安装较为费时，可以稍作休息"
sleep 3

unzip -q ${resource_path} -d ${es_path}

sysctl_conf='/etc/sysctl.conf'


# 在系统配置文件中找不到vm.max_map_count
if cat ${sysctl_conf} | grep "vm.max_map_count" > /dev/null; then
	printGreenText "在配置文件中已经存在vm.max_map_count"
else
	printGreenText "在配置文件中添加vm.max_map_count值"
	echo 'vm.max_map_count=655360'>>${sysctl_conf}
	
fi
sysctl -p

# 添加elasticsearch用户和用户组,必须先移除用户，再移除用户组
if cat /etc/group | grep "elasticsearch" > /dev/null; then
	printRedText "系统中已经存在elasticsearch用户组"
else
	groupadd elasticsearch
	
fi 

if cat /etc/passwd | grep "elasticsearch" > /dev/null; then
	printRedText "系统中已经存在elasticsearch用户"
else
	printGreenText "创建elasticsearch用户"
	useradd elasticsearch -g elasticsearch -p elasticsearch
	
fi 


# 对指定文件夹指定用户和用户组
chown -R elasticsearch:elasticsearch ${es_path}/deploy_aarch64/search/elasticsearch-5.4.0/
chown -R elasticsearch:elasticsearch ${es_path}/deploy_aarch64/logs/elasticsearch/
chown -R elasticsearch:elasticsearch ${es_path}/deploy_aarch64/data/elasticsearch/

# 对指定文件夹修改权限
chmod -R 755 ${es_path}/deploy_aarch64/search/elasticsearch-5.4.0/bin/
chmod -R 755 ${es_path}/deploy_aarch64/search/elasticsearch-5.4.0/config/
chmod -R 755 ${es_path}/deploy_aarch64/search/elasticsearch-5.4.0/lib/
chmod -R 755 ${es_path}/deploy_aarch64/search_config
chmod -R 755 ${es_path}/deploy_aarch64/service

echo "searchx_config_path: ${searchx_config_path}"
echo "manager_ip: ${manager_ip}"
# 将searchx的配置文件放到指定位置
if [ ! -e ${searchx_config_path} ]; then
    printYellowText "searchx的配置文件路径为: ${searchx_config_path}"
    mkdir -p ${searchx_config_path}
fi
cp ${es_path}/deploy_aarch64/search_config/* ${searchx_config_path}

printGreenText "修改searchx配置"
sed -i "s/^.*es\.network:.*$/es\.network: ${manager_ip}/" ${searchx_config_path}/search.yml
cat ${searchx_config_path}/search.yml | grep "es.network"

printGreenText "修改elasticsearch配置"

es_config_file=${es_path}/deploy_aarch64/search/elasticsearch-5.4.0/config/elasticsearch.yml

sed -i -e "s/^network\.host:.*$/network\.host: ${manager_ip}/" ${es_config_file}
sed -i -e "s|^.*path\.data:.*$|path\.data: ${es_path}/deploy_aarch64/data/elasticsearch|" ${es_config_file}
sed -i -e "s|^.*path\.logs:.*$|path\.logs: ${es_path}/deploy_aarch64/logs/elasticsearch|" ${es_config_file}

# egrep在脚本环境执行下，无法查找到命令
cat ${es_config_file} | grep -E "data|logs|host"

# 将service文件放到指定位置
cp ${es_path}/deploy_aarch64/service/* ${es_service_path}
systemctl daemon-reload

printGreenText "elasticsearch和searchx平台安装成功"
echo
echo
