#!/bin/bash

CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M:%S")
echo $CURRENT_TIME
mkdir -p ./output/$CURRENT_TIME

ENV=dsw

MODEL_SIZE=A3B                    # 模型大小: 7B, 14B, 32B, A3B
WORLD_SIZE=32                     # 训练使用的GPU卡总数,要和hostfile中的GPU总数一致
MICRO_BATCH_SIZE=1                # 每个PP的微批量大小
NUM_MICROBATCHES=128              # 每个迭代的微批量数量  
LR=1e-5                           # 学习率
MIN_LR=1e-6                       # 最小学习率
SEQ_LEN=32768                     # 序列长度 32768, 4096
PAD_LEN=${SEQ_LEN}                # 填充长度
PR=bf16                           # 训练精度: fp16, bf16, fp8
TRAIN_ITERS=${TRAIN_ITERS:-4300}  #
MOE_AUX_LOSS_COEFF=0.005          #
MOE_Z_LOSS_COEFF=0.002            #

TP=1                              # 模型并行度
PP=8                              # 流水并行度
CP=1                              # 上下文并行度
ETP=1                             # 专家张量并行度
EP=4                              # 专家模型并行度
SP=true                           # 是否使用序列并行: true, false
DO=true                           # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=True                           # 是否优先使用Flash Attention: true, false
SFT=false                         # 是否执行微调训练: true, false
AC=full                           # 激活检查点模式: sel, full, offload, false
MP_PP0_LAYERS="none" # 12
MP_PP_LAST_LAYERS="none" # 13
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-block}
XMLIR_PARALLEL_SAVE_MEMORY=true
MP_AC_LAYERS=${MP_AC_LAYERS:-15}  # 重计算层数
OPTIMIZER_OFFLOAD=false           # 是否启用Offload optimizer: false, 或输入0～1的小数作为参数offload比例
SAVE_INTERVAL=3                 # 保存ckpt的间隔

#DATASET_PATH="/mnt/si0003568lza/default/train_test/yehua/dataset/llama2_dataset/llama_00_text_document"              # 训练数据集路径
DATASET_PATH="/mnt/seed-program-nas/001688/caizhi/qwen-datasets/mmap_qwen3_datasets_text_document"              # 训练数据集路径
#DATASET_PATH="/mnt/seed-program-nas/001688/caizhi/03-CPT-data/1005_scaled_processed/mmap_qwen2_datasets_text_document"              # 训练数据集路径
#DATASET_PATH="/mnt/seed-program-nas/001688/caizhi/zj_hangtian_data/data/03-CPT-data/1005_scaled"              # 训练数据集路径
VALID_DATASET_PATH=${DATASET_PATH}        # 验证数据集路径
#PRETRAIN_CHECKPOINT_PATH="/mnt/seed-program-nas/001688/libingqiang/Qwen/Qwen3-30B-A3B"     # 预训练模型路径
PRETRAIN_CHECKPOINT_PATH="/mnt/seed-program-nas/001688/caizhi/qwen3_train/hf_mcore_ckpt"     # 预训练模型路径
#PRETRAIN_CHECKPOINT_PATH="/mnt/seed-program-nas/001688/caizhi/qwen3_train/megatron-lm-musa-patch/examples/qwen3/Qwen3-30B-A3B-Base-TP4-PP4-EP2-test/"     # 预训练模型路径
MP_SFT_PACKING=false
CPT_CONTINUE=${CPT_CONTINUE:-false}

(( DP_SIZE = $WORLD_SIZE / ($TP * $PP) ))
echo $DP_SIZE
(( GLOBAL_BATCH_SIZE = $MICRO_BATCH_SIZE * $NUM_MICROBATCHES * $DP_SIZE ))
echo $GLOBAL_BATCH_SIZE

#((TRAIN_TOKENS_OR_ITERS = 10000 * $GLOBAL_BATCH_SIZE * $SEQ_LEN))     # 训练TOKEN或者Iter数
#((WARMUP_TOKENS_OR_ITERS = 100 * $GLOBAL_BATCH_SIZE * $SEQ_LEN))      # 预热TOKEN或者Iter数
OUTPUT_BASEPATH="./output_mcore_qwen3_pretrain"           # 训练输出日志文件路径

#echo $TRAIN_TOKENS_OR_ITERS
#echo $WARMUP_TOKENS_OR_ITERS

set -u
  WORK_HOME="$PWD"
  PATCH_HOME="$PWD"/../..
  EXPNAME="tp${TP}_pp${PP}_ep${EP}_dp${DP_SIZE}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}_Iters${TRAIN_ITERS}"
  HOSTFILE=./hostfile
  LOG_FILE=./output/$CURRENT_TIME/$EXPNAME.log
  SCRIPT_FILE=./run_qwen3_moe_zj.sh
set +u
RDZV_ID=${CURRENT_TIME}

COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
hostlen=$(cat $HOSTFILE | wc -l )

for host in ${hostlist[@]}; do
  echo $host
  cmd="bash -c 'cd $WORK_HOME; \
     bash $SCRIPT_FILE $ENV \
                       $MODEL_SIZE \
                       $MICRO_BATCH_SIZE \
                       $GLOBAL_BATCH_SIZE \
                       $LR \
                       $MIN_LR \
                       $SEQ_LEN \
                       $PAD_LEN \
                       $PR \
                       $MOE_AUX_LOSS_COEFF \
                       $MOE_Z_LOSS_COEFF \
                       $TP \
                       $PP \
                       $CP \
                       $ETP \
                       $EP \
                       $SP \
                       $DO \
                       $FL \
                       $SFT \
                       $AC \
                       $MP_PP0_LAYERS \
                       $MP_PP_LAST_LAYERS \
                       $RECOMPUTE_METHOD \
                       $MP_AC_LAYERS \
                       $OPTIMIZER_OFFLOAD \
                       $SAVE_INTERVAL \
                       $DATASET_PATH \
                       $VALID_DATASET_PATH \
                       $PRETRAIN_CHECKPOINT_PATH \
                       $MP_SFT_PACKING \
                       $CPT_CONTINUE \
                       $OUTPUT_BASEPATH \
                       $WORK_HOME \
                       $PATCH_HOME \
                       $EXPNAME \
                       $TRAIN_ITERS \
                       $CURRENT_TIME \
                       $HOSTFILE \
		       $host \
                       $RDZV_ID"

  # ssh -f -n -p 62218 $host "pip install datasets" 
  cmd_ssh=$cmd" > $LOG_FILE.$COUNT.$host 2>&1'"
  echo $cmd_ssh
  ssh -f -n $host $cmd_ssh
  ((COUNT++))
done
