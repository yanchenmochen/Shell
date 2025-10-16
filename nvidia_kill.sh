#!/bin/bash

# 检查参数
if [ -z "$1" ]; then
  echo "用法: $0 [python|vllm|all]"
  exit 1
fi

case "$1" in
  python)
    pattern="python"
    ;;
  vllm|VLLM)
    pattern="VLLM"
    ;;
  all)
    pattern="python|VLLM"
    ;;
  *)
    echo "无效参数: $1"
    echo "可选项: python | vllm | all"
    exit 1
    ;;
esac

# 查找并杀掉对应进程
pids=$(nvidia-smi | grep -E "$pattern" | awk '{print $5}' | grep -E '^[0-9]+$')

if [ -z "$pids" ]; then
  echo "未找到匹配 [$pattern] 的 GPU 进程。"
else
  echo "即将终止以下进程: $pids"
  echo "$pids" | xargs -r -P 4 -I {} kill -9 {}
  echo "已清理完毕。"
fi
