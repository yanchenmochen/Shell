#!/bin/bash

docker_orig=/usr/bin/docker

# 检查第一个参数是否为 "run"
if [ "$1" != "run" ]; then
  # 如果不是 "run"，直接传递所有参数给 docker
  "${docker_orig}" "$@"
  exit 0
fi

# 预定义的有效姓名数组
valid_names=(
    "yangfei"
    "zhangweixing"
    "chensheng"
    "jianglinnan"
    "dengqiuliang"
    "wanglouping"
    "cairunze"
    "lihaiyan"
    "yanjun"
    "zhangyi"
    "sunning"
    "chensi"
    "songquanheng"
    "duchengyao"
    "chenqun"
    "cuixin"
    "huangjunhao"
    "luchenhao"
    "dongjie"
    "wangyuanyuan"
    "wangfangyu"
    "fukejie"
    "lufanfeng"
    "wangzeyue"
    "wangyuyang"
    "zhangrui"
    "panshu"
)  

# 变量用于保存容器名称
container_name=""

# 遍历所有参数，查找 --name 参数并获取容器名称
for ((i=1; i <= $#; i++)); do
  if [ "${!i}" == "--name" ]; then
    # 获取 --name 后的参数，即容器名称
    next=$((i+1))
    container_name=${!next}
    break
  fi
done

# 如果未找到容器名称，提前退出
[ -n "$container_name" ] || { echo "Error: No container name specified with --name."; exit 1; }

# 分割容器名称，获取姓名部分
IFS='-' read -r name_part desc_part <<< "$container_name"

# 校验姓名部分是否在有效姓名数组中
# 校验容器名称格式，必须为 "姓名全拼-名称描述" 或 "name-description" 的格式
# 校验姓名部分是否在有效姓名数组中
name_valid=false
for valid_name in "${valid_names[@]}"; do
  if [ "$name_part" == "$valid_name" ]; then
    name_valid=true
    break
  fi
done

# 如果姓名不在有效数组中，输出错误并退出
if [ "$name_valid" == "false" ]; then
  echo "Error: The name part of the container name must be a valid name in the predefined list."
  echo "Error: Container name must follow the format 'namequanpin-container_description'"
  echo "for example, songquanheng-vllm-smoothquant is valid"
  exit 1
fi



echo "The container to start is ${container_name}"
echo "The start process will automatically set Huggingface and pip configuration, also, mount necessary nas directories"
echo "The specific value of configuration can be found in /mnt/nas_v1/common/public/config/docker.env"
echo "The start process slow because of the mount of two nas directories and parameters resolving"
echo "please wait for a moment..."

# 添加通用的环境变量文件和挂载
echo "The ultimate parameters is $@"

args=("$@")
# 获取除第一个参数外的所有参数
remaining_args=("${args[@]:1}")

"$docker_orig" run --name ${container_name} --env-file /mnt/nas_v1/common/public/config/docker.env --gpus all -v /mnt/nas_v1/common/public:/public -v /mnt/self-define:/mnt/self-define "${remaining_args[@]}" 
