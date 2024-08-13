#!/bin/sh
# crontab -e  */1 * * * * su - root -c "/usr/shell/schedule_pbs.sh"

get_pid() {
        echo $(netstat -anop | grep ":$1" | grep -w LISTEN | tr -s " " | awk -F' ' '{print $7}' | cut -d/ -f1 | sort | uniq)
}

print_date() {
        echo $(date "+%Y-%m-%d %H:%M:%S")
}

cd $(dirname $0)
pwd

. ../util/util_function.sh
. ../conf/camp.conf

printGreenText "log_dir=${daemon_log_dir}"
log_dir=${daemon_log_dir}

log_path=${log_dir}/pbs/cron/cron.log
log_size=$(du -k "$log_path" | awk '{print $1}')

if [ ${log_size} -gt 10240 ] ;then
        mv ${log_path} "/opt/pbs/cron/cron-$(print_date).log"
fi

if [ ! -e ${log_dir}/pbs/cron/ ]; then
        mkdir -p ${log_dir}/pbs/cron/
fi

echo "$(print_date) do the shell" >> ${log_path}

echo >> ${log_path}
pbs_mom_pid=$(get_pid 15002)
if [ -z ${pbs_mom_pid} ] ; then
        echo "15002 15003 are pbs_mom ports" >> ${log_path}
        echo "PBS mom mode... fail" >> ${log_path}
    echo "start pbs_mom  $(print_date)" >> ${log_path}
        chmod 0555 /var/
        pbs_mom
        startPbs="true"
        echo "start pbs_mom over" >> ${log_path}
fi

echo >> ${log_path}
if [ ! -z ${startPbs} ] ; then
        echo "PBS trqauthd is ready" >> ${log_path}
    echo "start trqauthd $(print_date)" >> ${log_path}
        trqauthd
        echo "start trqauthd over" >> ${log_path}
fi

#echo >> ${log_path}
#pbs_mom_pid=$(get_pid 15002)
#if [ -z ${pbs_mom_pid} ] ; then
#        echo "15002 15003 are pbs_mom ports" >> ${log_path}
#        echo "PBS mom mode... fail" >> ${log_path}
#    echo "start pbs_mom  $(print_date)" >> ${log_path}
#        pbs_mom
#        startPbs="true"
#        echo "start pbs_mom over" >> ${log_path}
#fi
