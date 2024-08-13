#!/bin/bash

# 定义帮助函数
show_help() {
    echo "Usage: $0 [OPTIONS] CONTAINER_NAME..."
    echo
    echo "Options:"
    echo "  --help, -h      Display this help message."
    echo
    echo "Description:"
    echo "  This script restarts Docker containers and their SSH services."
    echo "  If no container names are provided, it uses the default list."
}

# 定义默认容器名称数组
container_names=("vllm-smoothquant" "smoothquant" "lmdeploy050")

# 检查是否提供了帮助选项
if [[ "$@" == *"--help"* ]] || [[ "$@" == *"-h"* ]]; then
    show_help
    exit 0
fi

# 如果提供了命令行参数，则使用这些参数作为容器名称
if [ "$#" -gt 0 ]; then
    # 使用 $@ 获取所有传入的参数，并覆盖默认的容器名称数组
    container_names=("$@")
fi

# 遍历容器名称并执行操作
for container_name in "${container_names[@]}"; do
    docker restart "$container_name"
    docker exec -it "$container_name" /etc/init.d/ssh restart
done