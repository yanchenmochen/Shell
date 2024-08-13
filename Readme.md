# <center>脚本使用指南</center>

# 简介

在为贺ls项目4零1保障智能分析一体机时，由于需要频繁的修改智能分析服务器的的ip地址，因此在现场撰写了该脚本，shell目录用于一键修改智能分析服务器的ip地址。

# 目录简介

```c

```



其中conf/camp.conf比较重要

```properties
# 服务器所在的ip，实际运行时服务管理、应用管理、数据库、es、searchx均使用此ip# 
server_ip=9.9.9.245
# iSC相关ip和配置
# 此ip必须填写，不能为空
isc_ip=9.9.9.245
# 此处appkey和appsecret需要iSC平台运行管理中心-状态监控-API网关-参数配置-API管理
appkey=bbb
appsecret=cccc
```

上图中文件已做了相关的注释，不再赘述

# 使用指南

在shell脚本组使用过程中，使用的步骤如下：

## 配置conf/camp.conf

使用真实的智能分析服务ip地址、iSC ip地址、接口接入的appkey和appsecret来配置conf/camp.conf文件。

## 执行

当现场运维人员根据现场的ip划分已经成功的配置好**conf/camp.conf**，则可以一键执行下面的程序

```shell
cd shell/
./changeIp.sh
```

## 修改内容

修改的文件列表所示：

```shell
vim /opt/tomcat/webapps/ROOT/WEB-INF/classes/config/application-dev.properties
vim /usr/tomcat/webapps/platform/WEB-INF/classes/application.properties
vim /hikmars3/system/logs/config.json
vim /etc/elasticsearch/elasticsearch.yml
vim /etc/usearch/search.yml
```

