# shell目录中脚本使用说明

## 简介

shell文件夹中脚本主要用于在一台新服务器上部署资源智能识别与分析服务。

脚本中的内容有：

1、FT+24个A200一体机初始化相关脚本；

2、部署相关环境及程序；

环境：Java、node、nginx、tomcat

程序：camp程序包（前后端程序）、platform程序包、设备发现程序包、运管中心程序包（前后端程序）

3、配置IP；

4、定时任务配置；

5、运维相关脚本：portpid、portkill、campstat

## 使用说明

### 一体机A200初始化配置

一体机的初始化需要完成以下几个步骤：

1、修改对应网卡配置，修改A200板子IP，并搭建网桥

```
本步骤需手动修改
```

2、服务器上调用脚本，安装expect自动应答程序

```shell
# 进入shell目录下，执行下面命令，服务器上执行
sh deploy_expect.sh
```

3、服务器上调用脚本，批量修改A200板子配置，允许root用户远程登录

```shell
# 进入shell目录下，执行下面命令，服务器上执行
sh permit_root_login_file.sh
```

4、服务器上调用脚本，批量推公钥，允许用户免密登录

```shell
# 进入shell目录下，执行下面命令，服务器上执行
sh batch_push_pub.sh
```

5、服务器上调用脚本，批量拷贝shell目录到A200板子上

```shell
执行脚本copy_shell_dir_to_a200.sh  
# 进入shell目录下，执行下面命令，服务器上执行
sh copy_shell_dir_to_a200.sh
```

6、服务器上调用脚本，批量移除A200虚拟网卡usb0的信息，并且修改域名

```shell
# 进入shell目录下，执行下面命令，服务器上执行,该脚本会重启a200
sh batch_pre_configure_a200.sh
```

7、A200板子上安装Java环境

```shell
脚本batch_deploy_java.sh  批量给A200板子部署JDK环境
# 进入shell目录下，执行下面命令，服务器上执行
sh batch_deploy_java.sh
```

8、安装PBS服务、配置定时任务

```shell
批量部署
# 进入shell目录下，执行下面命令，服务器上执行
sh batch_a200_configure.sh
```



### FT服务器程序部署

以下部署指资源智能识别平台上传应用部署，包括campus前后端程序、运管中心前端程序、ES相关程序

#### 1、修改脚本配置文件

编辑shell/conf目录下的camp.conf文件

```
该文件提供shell目录中一些脚本的配置，包括程序安装路径、服务IP等配置等。
```



#### 2、部署程序

执行shell目录下的deploy_all.sh

```
该脚本作用：依次部署tomcat、后端campus、node、nginx、campus_web、ES程序
```



#### 3、修改IP

程序部署好以后，执行shell目录下的changeIp.sh

```shell
#脚本作用：该脚本用于一键修改401项目相关的服务ip，其中修改的内容包括
#  1. 模块管理
#     1. camp项目的ip
#     2. camp项目中数据库所在的ip，该ip与camp的ip相同
#     3. hikinfer.ip: 服务管理所在的ip
#     4. network.host: elasticsearhsearch的ip
#     5. es.network: search 检索平台的ip地址
#  2. iSC相关
#     tomcat.ip: isc取流服务所在的服务器ip
#     1. isc_ip: isc服务器所在的ip
#     2. appkey: isc接口接入时需要传入的appkey, 在运行管理中心-状态监控-API网关-参数配置-API管理
#     3. appsecret: isc接口接入时需要传入的appsecret

#  3. 推理服务模块
#     1. 该部分对应文件/hikmars3/system/logs/config.json，直接删掉，然后添加即可
```

==注意：该脚本修改ip后，会重启服务，需稍等片刻==



#### 4、添加定时任务

执行shell/module目录下的add_manager_application_schedule_task.sh脚本文件

```
# 警告：管理节点的/etc/network/interfaces中仅有一行address ip,该脚本才能正确工作。
# 脚本作用： 该脚本用于把定时运维脚本module/manager_schdule_task.sh
#        自动加入管理节点crontab服务中
# 8090 前端主程序
# 8080 后端主程序
# 5000 服务管理程序
# 15001 pbs_server端口
# 15002 pbs_mom框架程序端口
# 15003 注意：pbs_mom占据两个端口
# 15004 pbs_sched框架程序端口
# 9200  es数据检索与外部通讯端口
# 9300  数据检索节点内部通讯端口
# 9400  检索平台端口
# 5236  数据库端口
# 8888  node端口，实时预览
# 1937  视频拼接使用端口
```



