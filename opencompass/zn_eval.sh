#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/local/openmpi/lib
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

python run.py --models hf_llama3_8b_instruct --datasets winogrande_5shot_gen_b36770 --max-num-workers 8 --batch-size 4 --work-dir "/mnt/seed-program/001688/zn/opencompass/outputs/winogrande_5shot_gen_b36770_oneshot" --reuse latest

#python run.py --models hf_llama3_1_8b --datasets mmlu_gen --hf-num-gpus 8 --max-num-workers 8 --batch-size 4 --reuse latest
# /mnt/seed-program/001688/models/llama2-7b-hf
#python run.py --hf-type base --hf-path  /mnt/seed-program/001688/zn/opencompass/opencompass/configs/models/hf_llama/lmdeploy_llama3_1_8b.py --tokenizer-path  /mnt/seed-program/001688/models/Meta-Llama-3.1-8B --batch-size 8 --max-seq-len 4096 --datasets mmlu_gen
# python run.py --hf-type base --hf-path  /mnt/seed-program/001688/models/llama2-7b-hf --tokenizer-path  /mnt/seed-program/001688/models/llama2-7b-hf --batch-size 8 --max-seq-len 2048 --datasets mmlu_gen --debug
