#!/bin/bash

cd $(dirname $0)
# 定义源文件路径前缀
source_type=$1
SOURCE_FILE_SUFFIX=".list"

# 读取 /etc/os-release 文件
source /etc/os-release

# 检查是否为 Ubuntu 并获取版本号
if [[ "$ID" == "ubuntu" ]]; then
    UBUNTU_VERSION=$VERSION_ID
    echo "Detected Ubuntu version: $UBUNTU_VERSION"
else
    echo "This script is intended for Ubuntu systems only."
    exit 1
fi

# 检查版本号是否符合要求
case $VERSION_ID in
    18.04|20.04|22.04)
        ;;
    *)
        echo "Unsupported Ubuntu version. This script supports 18.04, 20.04, and 22.04."
        exit 1
        ;;
esac


# 根据用户的选择设置源文件名
case $source_type in
    1)
        SOURCE_FILE="source/qinghua-ubuntu-$VERSION_ID.${SOURCE_FILE_SUFFIX}"
        ;;
    2)
        SOURCE_FILE="source/ali-ubuntu-$VERSION_ID.${SOURCE_FILE_SUFFIX}"
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

# 检查并备份 sources.list
if [ -f /etc/apt/sources.list ]; then
    sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
    echo "sources.list has been backed up to sources.list.bak"
else
    echo "No existing sources.list file found."
fi

# 检查并备份 /etc/apt/sources.d/ 目录下的文件
if [ -d /etc/apt/sources.d/ ]; then
    for file in /etc/apt/sources.d/*; do
        if [ -f "$file" ]; then
            sudo cp "$file" "${file}.bak"
            echo "Backed up $file to ${file}.bak"
        fi
    done
else
    echo "/etc/apt/sources.d/ directory does not exist or is empty."
fi

# 拷贝新的源文件到 /etc/apt/
sudo cp "$SOURCE_FILE" /etc/apt/sources.list

# 更新 apt 缓存
sudo apt update

echo "APT source configuration completed."