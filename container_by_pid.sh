#!/bin/bash

# 表头
printf "%-15s %-12s %-30s %-30s\n" "in_container" "容器ID" "容器名称" "镜像名称"

# 遍历所有输入的 PID
for PID in "$@"
do
  # 检查PID是否存在
  if [ -z "$PID" ]; then
    echo "请提供PID"
    exit 1
  fi

  # 检查PID是否在容器中运行，并提取第一个出现的容器ID
  CGROUP=$(cat /proc/$PID/cgroup | grep 'docker')
  CONTAINER_ID_LONG=$(echo "$CGROUP" | grep -oP '(?<=/docker/)[a-f0-9]{64}' | head -n 1)

  if [ -n "$CONTAINER_ID_LONG" ]; then
    # 提取短ID（前12个字符）
    CONTAINER_ID_SHORT=${CONTAINER_ID_LONG:0:12}

    # 获取容器名称和镜像名称
    CONTAINER_NAME=$(docker inspect --format '{{.Name}}' "$CONTAINER_ID_LONG" | sed 's/^.\{1\}//')
    IMAGE_NAME=$(docker inspect --format '{{.Config.Image}}' "$CONTAINER_ID_LONG")

    printf "%-15s %-12s %-30s %-30s\n" "Yes" "$CONTAINER_ID_SHORT" "$CONTAINER_NAME" "$IMAGE_NAME"
  else
    printf "%-15s %-12s %-30s %-30s\n" "No" "N/A" "N/A" "N/A"
  fi
done
