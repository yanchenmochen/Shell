#!/bin/bash
dir=$1
cd "$dir"

# 定义一个数组，用于存储所有 dataType
dataTypes=()

# 遍历目录下所有 .json 文件
for file in *.json; do
  # 使用 jq 提取 dataType 字段，追加到 dataTypes 数组
  dataTypes+=($(jq -r '.[].dataType' "$file"))
done

# 对 dataTypes 数组中的所有值排序并去重
uniqueDataTypes=($(echo "${dataTypes[@]}" | tr ' ' '\n' | sort -u))

# 输出所有唯一的 dataType 值
echo "唯一的 dataType 值有："
for dataType in "${uniqueDataTypes[@]}"; do
  echo "$dataType"
done
