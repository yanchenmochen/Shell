#!/bin/bash

# 定义参数为脚本文件路径
script_file="$1"

# 根据文件扩展名选择解释器
if [[ "$script_file" == *.sh ]]; then
    interpreter="bash"
elif [[ "$script_file" == *.py ]]; then
    interpreter="python"
else
    echo "Unsupported file type: $script_file"
    exit 1
fi

# 循环执行直到成功
while true; do
    $interpreter "$script_file" && break || echo "Command failed, retrying..."
done