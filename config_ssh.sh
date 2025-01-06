#!/bin/bash

# 定义变量
SSH_PACKAGE="openssh-server"
SSH_CONFIG_FILE="/etc/ssh/sshd_config"

# 接收端口号和密码作为参数
SSH_PORT="22"  # 第一个参数为端口号
PASSWORD="tianshu@123"  # 第二个参数为密码

# 改变工作目录到脚本所在目录
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

# 检查 openssh-server 是否已安装
if ! dpkg -s $SSH_PACKAGE > /dev/null 2>&1; then
    bash ./config_apt_source.sh
    echo "Installing $SSH_PACKAGE..."
    apt-get update -y
    apt-get install $SSH_PACKAGE -y
fi

# 备份原有配置文件
cp $SSH_CONFIG_FILE $SSH_CONFIG_FILE.bak

# 修改 SSH 配置文件
echo "Port $SSH_PORT" | tee -a $SSH_CONFIG_FILE > /dev/null
echo "PermitRootLogin yes" | tee -a $SSH_CONFIG_FILE > /dev/null

# 设置 root 密码
echo "root:$PASSWORD" | chpasswd

# 设置我的公钥信息
echo "configure public key"
bash save_pub.sh

# 重启 SSH 服务
service ssh restart

# 输出完成信息
echo "SSH server configured on port $SSH_PORT and password set."
