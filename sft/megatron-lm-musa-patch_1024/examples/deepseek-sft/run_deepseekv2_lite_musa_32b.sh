#!/bin/bash

CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M:%S")
echo $CURRENT_TIME
mkdir -p ./output/$CURRENT_TIME

TP_SIZE=1
PP_SIZE=2
EP_SIZE=8
WORLD_SIZE=16
MICRO_BATCH_SIZE=1
NUM_MICROBATCHES=128
(( DP_SIZE = $WORLD_SIZE / ($TP_SIZE * $PP_SIZE) ))
echo $DP_SIZE
(( GLOBAL_BATCH_SIZE = $MICRO_BATCH_SIZE * $NUM_MICROBATCHES * $DP_SIZE ))
echo $GLOBAL_BATCH_SIZE

set -u
  WORK_HOME="$PWD"
  PATCH_HOME="$PWD"/../..
  PATCH_HOME=/mnt/seed-program-nas/001688/haoran.huang/megatron-lm-musa-patch_1014
  EXPNAME="tp${TP_SIZE}_pp${PP_SIZE}_dp${DP_SIZE}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}"
  DATA_PATH=/mnt/seed17/001688/launch.zgc/repo/Megatron-LM/tulu3_sft_mixture_00005_text_document
  HOSTFILE=./hostfile
  LOG_FILE=./output/$CURRENT_TIME/$EXPNAME.log
  TOKENIZED_MODEL=/mnt/seed-program-nas/001688/xiechunhong/1B_Loss_Align/Meta-Llama-3-tokenizer
  SCRIPT_FILE=./deepseek-v2-lite/run_pretrain_deepseekv2_musa_32b.sh
  RDZV_ID=$CURRENT_TIME
set +u

cmd="bash -c 'cd $WORK_HOME; \
     bash $SCRIPT_FILE $WORK_HOME $PATCH_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" \
     $TP_SIZE $PP_SIZE $EP_SIZE \
     $MICRO_BATCH_SIZE $GLOBAL_BATCH_SIZE $TOKENIZED_MODEL $RDZV_ID"

COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
hostlen=$(cat $HOSTFILE | wc -l )

for host in ${hostlist[@]}; do
    ssh $host "pkill -f '/opt/conda/envs/py310/bin/torchrun'" 
    echo "$host is killed."
done
# cmd_ssh=$cmd" > $LOG_FILE.$COUNT.$host 2>&1'"
# eval $cmd_ssh
COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
  cmd_ssh=$cmd" > $LOG_FILE.$COUNT.$host 2>&1'"
  # cmd_ssh=$cmd" '"
  echo $cmd_ssh
  ssh -f -n $host $cmd_ssh
  # echo $host, "bash -c 'cd $FlagScale_HOME/megatron; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  # ssh -f -n $host "bash -c 'cd $FlagScale_HOME/megatron; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  ((COUNT++))
done

