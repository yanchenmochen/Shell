#!/bin/bash

docker_orig=/usr/bin/odocker

# 检查第一个参数是否为 "run"
if [ "$1" != "run" ]; then
  # 如果不是 "run"，直接传递所有参数给 docker
  "${docker_orig}" "$@" 2>&1
  exit $?
fi

# 检查 --name 是否使用了 "=" 作为分隔符
for arg in "$@"; do
  if [[ "$arg" =~ --name=.* ]]; then
    echo "Error: Invalid parameter format. Please use '--name <container_name>' instead of '--name=<container_name>'."
    exit 1
  fi
done

# 从文件读取有效姓名列表，并去除前后空格
valid_names_file="/mnt/nas_v1/common/public/config/valid-names"
valid_names=()

# 读取文件中的有效姓名到数组
while IFS= read -r line; do
  # 去除前后空格
  name=$(echo "$line" | xargs)
  valid_names+=("$name")
done < "$valid_names_file"
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
  echo
  echo "Valid names are:"
  for valid_name in "${valid_names[@]}"; do
    echo "  - $valid_name"
  done
  exit 1
fi

# 要映射到容器的端口
container_port=22

# 查找未被占用的端口范围
start_port=10000
end_port=20000

# 查找未占用的端口
for ((port=$start_port; port<=$end_port; port++)); do
  # 检查端口是否被占用
  if ! ss -tuln | grep -q ":$port "; then
    # 找到未占用的端口，输出并退出
    echo "Selected host port: $port"
    host_port=$port
    break
  fi
done

# 如果没有找到空闲端口，退出脚本
if [ -z "$host_port" ]; then
  echo "No available port found in range $start_port-$end_port."
  exit 1
fi


echo "We create this command to share model、datasets、and downloads of other workers, speeding up the creation of environment"

echo "The start process will automatically set Huggingface and pip configuration, also, mount necessary nas directories"
echo "default huggingface models directory is /mnt/nas_v1/common/public/model"
echo "default huggingface datasets directory is /mnt/nas_v1/common/public/dataset"
echo
echo "default pip source is https://pypi.tuna.tsinghua.edu.cn/simple"
echo "default pip cache is /mnt/nas_v1/common/public/pip_cache"
echo
echo "The specific value of configuration can be found in /mnt/nas_v1/common/public/config/docker.env"

echo 
echo "of course you can still use the original docker command which is named odocker"

echo "The start process is slow because of the mount of two nas directories and parameters resolving"
echo "please wait for a moment..."

# 添加通用的环境变量文件和挂载
echo "The container to start is ${container_name}"
echo "The ultimate parameters are $@"

args=("$@")
# 获取除第一个参数外的所有参数
remaining_args=("${args[@]:1}")

"$docker_orig" run --name ${container_name} --env-file /mnt/nas_v1/common/public/config/docker.env --gpus all -v /mnt/nas_v1/common/public:/public -v /mnt/self-define:/mnt/self-define -p $host_port:$container_port "${remaining_args[@]}" 2>&1
