#!/bin/bash
# author：songquanheng
# 2020年12月24日09:34:59
# 脚本用途： 该脚本用于一键卸载hjj程序
# 1、从camp.conf获取配置信息
# 2、卸载es
# 3、卸载web 
# 4、卸载nginx
# 5、卸载node
# 6、卸载campus后台程序

cd $(dirname $0)


. util/util_function.sh
. conf/camp.conf
printGreenText "当前管理节点IP: ${manager_ip}"
if test $(whoami) != 'root'; then
	printRedText "请使用root用户执行一键卸载操作"
	exit 4
fi

bash module/undeploy_symbolic.sh

# 卸载elasticsearch和search
# bash module/undeploy_es.sh

bash module/undeploy_campus_web.sh

bash module/undeploy_nginx.sh

bash module/undeploy_node.sh

bash module/undeploy_campus_jar.sh

# bash module/undeploy_discovery.sh

bash module/undeploy_monitor.sh

bash module/undeploy_algorithm.sh

bash module/undeploy_iot.sh

bash module/undeploy_net_config.sh
printGreenText "程序卸载完成，还你干净无染服务器"

