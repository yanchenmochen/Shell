#!/bin/bash

# 检查是否提供了仓库URL参数
if [ -z "$1" ]; then
  echo "请提供仓库的URL，例如：./clone_without_weights.sh <仓库URL>"
  exit 1
fi

# 仓库URL
REPO_URL="$1"

# 定义克隆的默认目录，如果未设置 REPO_DIR 环境变量，则使用 /mnt/model
DEFAULT_DIR="/mnt/model"
CLONE_DIR="${REPO_DIR:-$DEFAULT_DIR}"

# 创建克隆目录（如果不存在）
mkdir -p "$CLONE_DIR"
cd "$CLONE_DIR" || exit

# 克隆仓库结构，但不下载文件内容
git clone --no-checkout --filter=blob:none "$REPO_URL"
cd "$(basename "$REPO_URL" .git)" || exit

# 初始化稀疏克隆
git sparse-checkout init --cone

# 配置稀疏克隆规则，排除权重文件
echo '/*' > .git/info/sparse-checkout
echo '!model*.safetensors' >> .git/info/sparse-checkout

# 拉取除权重文件以外的其他文件
git checkout main  # 如果是其他分支，请替换为对应的分支名

echo "仓库克隆完成，不包含模型权重文件"
echo "克隆目录: $CLONE_DIR/$(basename "$REPO_URL" .git)"

