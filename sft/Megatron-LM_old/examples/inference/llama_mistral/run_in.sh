HOSTFILE=/mnt/seed-program-nas/001688/haoran.huang/Megatron-LM_old/examples/inference/llama_mistral/hostfile
cat $HOSTFILE

WORK_HOME=/mnt/seed-program-nas/001688/haoran.huang/Megatron-LM_old/examples/inference/llama_mistral
SCRIPT_FILE=run_text_generation_ds32b.sh


COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
hostlen=$(cat $HOSTFILE | wc -l )

for host in ${hostlist[@]}; do
    echo $host
    ssh $host "pkill -9 python"
done

LOG_FILE=test.log

cmd="bash -c 'cd $WORK_HOME; \
     bash $SCRIPT_FILE"


COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
  cmd_ssh=$cmd" > $LOG_FILE.$COUNT.$host 2>&1'"
  echo $cmd_ssh
  ssh -f -n $host $cmd_ssh
  ((COUNT++))
done



