#!/bin/bash

# 参数检查
if [ $# -ne 3 ]; then
  echo "使用方法: $0 <基础IP> <起始IP> <终止IP>"
  echo "示例: $0 10.107.204 72 73"
  exit 1
fi

# 获取参数
BASE_IP="$1."
START_IP=$2
END_IP=$3

# SSH用户
USER="root"

# SSH密码
PASSWORD="qsgctys@05980"

# 公钥路径
PUB_KEY_PATH="$HOME/.ssh/id_rsa.pub"

# 检查sshpass是否安装
if ! command -v sshpass &> /dev/null; then
  echo "sshpass未安装，请先安装它。"
  exit 1
fi

# 检查公钥是否存在
if [ ! -f "$PUB_KEY_PATH" ]; then
  echo "SSH公钥未找到，请生成公钥或指定正确的路径。"
  exit 1
fi

# 循环遍历IP范围并复制公钥
for i in $(seq $START_IP $END_IP); do
  FULL_IP="$BASE_IP$i"
  echo "正在将公钥复制到 $USER@$FULL_IP..."
  
  # 使用sshpass传递密码并复制公钥
  sshpass -p "$PASSWORD" ssh-copy-id -i "$PUB_KEY_PATH" -o StrictHostKeyChecking=no "$USER@$FULL_IP"
  
  if [ $? -eq 0 ]; then
    echo "成功将公钥复制到 $FULL_IP"
  else
    echo "无法连接到 $FULL_IP，跳过..."
  fi
done

echo "所有操作完成。"

