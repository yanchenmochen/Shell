#!/bin/bash
TRUE="true" 
FALSE="false"
readonly TRUE
readonly FALSE

# 获取文件的绝对路径
absolutePath() {
        echo $(readlink -f "$1")
}

# 获取相对文件目录的绝对路径
absoluteDir() {
    cd "$1"
    echo $(pwd)
}

printGreenText() {
        echo -e  "\033[32m----$1----\033[0m"
}
printYellowText() {
        echo -e  "\033[33m----$1----\033[0m"
}

printRedText() {
        echo -e "\033[31m----$1----\033[0m"
}

printYellowBgGreenText() {
    echo -e  "\033[43;32m----你好----\033[0m"
}

contains() {
        result=$(echo $1 | grep "$2")
        if [ -n "${result}" ] ; then
                echo ${TRUE}
        else
                echo ${FALSE}
        fi
}

valid_ip() {
    domain_count=$(echo $1 | awk -F . '{print NF}')
    if [ ${domain_count} -ne 4 ]; then
        echo $FALSE
        exit 5
    fi
    ip=$1
    readonly ip
    ip_domains[0]=$(echo "${ip}" | awk -F . '{print $1}')
    ip_domains[1]=$(echo "${ip}" | awk -F . '{print $2}')
    ip_domains[2]=$(echo "${ip}" | awk -F . '{print $3}')
    ip_domains[3]=$(echo "${ip}" | awk -F . '{print $4}')

    for domain in ${ip_domains[@]}
    do
        if [ $(valid_ip_domain $domain) != ${TRUE} ]; then
            echo $FALSE
            exit 5
        fi
    done
    echo $TRUE
    exit 0
}

valid_ip_domain() {
    if [ $1 -ge 0 -a $1 -le 255 ]; then
        echo $TRUE
    else
        echo $FALSE
    fi 
}

trim() {
    : "${1#"${1%%[![:space:]]*}"}"
    : "${_%"${_##*[![:space:]]}"}"
    printf '%s\n'"$_"
}

# 查询某个端口所对应的进程pid
# 参数$1表示的是端口号
pid_of_port() {
	echo $(netstat -anop | grep -w $1 | grep -w LISTEN | tr -s " " | awk -F' ' '{print $7}' | cut -d/ -f1 | sort | uniq)
}

# 使用kill命令终止某个端口所对应的进程，其中参数$1表示的是端口号
portkill() {
	echo "将要终止的程序端口为$1"
	kill -9 $(pid_of_port $1)
}

ip() {
	echo $(cat /etc/network/interfaces | grep address | awk -F' ' '{print $2}')
}

# 该函数完成在文件中替换源串和目标串，其中包含三个参数
# 第一个参数为源串， 第二个参数为目标串， 第三个参数为文件路径
# 该函数主要用来处理
#db.schema=SSZK
#db.username=ADMIN
#这种形式的替换
replace() {
	sed -i "s|.*$1=.*$|$1=$2|g" $3
}

# 该函数完成在文件中替换源串和目标串，其中包含三个参数
# 第一个参数为源串， 第二个参数为目标串， 第三个参数为文件路径
# 该函数主要用来处理
# -Djava.rmi.server.hostname=10.100.135.7d -Dcom.sun.management.jmxremote.port=10007
#这种形式的替换
replace_hostname_in_service() {
    sed -i "s/$1=[0-9]*.[0-9]*.[0-9]*.[0-9]*/$1=$2/g" $3
}