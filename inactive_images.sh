#!/bin/bash

# 获取所有镜像 ID
all_images=$(docker images -q | sort | uniq)

# 获取所有容器正在使用的镜像 ID
active_images=$(docker ps -a --format '{{.Image}}' | xargs -n1 docker images -q | sort | uniq)

# 找出不活跃的镜像
inactive_images=$(comm -23 <(echo "$all_images") <(echo "$active_images"))

# 输出结果
if [ -z "$inactive_images" ]; then
    echo "No inactive images found."
else
    echo "Inactive images:"
    echo "$inactive_images"
fi

