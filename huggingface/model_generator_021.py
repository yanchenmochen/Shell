import torch
import torch_musa
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='模型生成器脚本')
    parser.add_argument('-d', '--debug', action='store_true', 
                        help='启用调试模式（对应DEBUG环境变量）')
    parser.add_argument('-s', '--save-pt', action='store_true',
                        help='启用保存PT文件（对应SAVE_PT环境变量）')
    parser.add_argument('-l', '--load-pt', action='store_true',
                        help='启用加载PT文件（对应LOAD_PT环境变量）')
    parser.add_argument('--use-mg', action='store_true',
                        help='使用MG模式（对应USE_MG环境变量）')
    parser.add_argument('--layer-diff', action='store_true',
                        help='启用层级操作差异对比（对应LAYER_OP_DIFF环境变量）')
    parser.add_argument('--moe-diff', action='store_true',
                        help='启用MoE操作差异对比（对应MOE_OP_DIFF环境变量）')
    return parser.parse_args()

# 解析命令行参数
args = parse_args()

# 根据命令行参数设置环境变量
if args.debug:
    os.environ['DEBUG'] = '1'
if args.save_pt:
    os.environ['SAVE_PT'] = '1'
    os.environ['ckpt_dir'] = '/mnt/seed-program-nas/001688/dongjie/X10000-1029/Megatron-LM/examples/inference/moe16b/output_hf_021'
if args.load_pt:
    os.environ['LOAD_PT'] = '1'
if args.use_mg:
    os.environ['USE_MG'] = '1'
if args.layer_diff:
    os.environ['LAYER_OP_DIFF'] = '1'
if args.moe_diff:
    os.environ['MOE_OP_DIFF'] = '1'

# 设置模型路径
# model_path = "/mnt/self-define/songquanheng/output-Llama3_1-8b-sft/checkpoint/mcore-llama3-1-8B-sft-iter3000"
# /mnt/hw-nas/002147/yanjun/data/pretrain/checkpoint/12.8T/tp1_pp2_ep8_mbs2_gbs4800-iter50000-hf 为021-32B转换出来的检查点
# /mnt/seed17/001688/honghong/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/moe32b/iter_0050000_hf_new honghogn转换检查点
# /mnt/seed17/001688/honghong/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/moe32b/iter_0050000_hf 
# model_path = "/mnt/seed-program-nas/001688/songquanheng/model/iter_0050000_hf_new"
model_path = "/mnt/seed-program-nas/001688/honghong/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/moe32b/iter_0030000_hf_1107"
# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
try:
    import torch_musa
    has_musa = True
except ImportError:
    has_musa = False

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

if has_musa:
    model = model.musa()
else:
    model = model.cuda()

# 确保 pad_token 有效
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 获取模型默认的生成配置
generation_config = GenerationConfig.from_pretrained(model_path)
generation_config.pad_token_id = tokenizer.pad_token_id  # 确保 pad_token_id 正确设置
generation_config.temperature = 1e-6
generation_config.top_k = 1
generation_config.top_p = 0

# 定义 prompts 列表
prompts = [
    "who are you?",
    # "what is the result of 1+1?",
    # "写一首关于夏天海边风景的短诗。",
    # "小王有5个苹果，给了小李2个，还剩多少？",
    # "请将下面的英文翻译成中文：‘Deep learning models require large amounts of data.’",
    # "从前有一位勇敢的探险家，他踏上了一段未知的旅程，然后……",
    # "Explain the difference between supervised and unsupervised learning.",
    # "Write a short poem about a rainy afternoon.",
    # "If John has 10 oranges and gives 3 to Mary, how many does he have left?",
    # "Translate the following sentence into Chinese: 'Artificial intelligence is transforming industries.'",
    # "Once upon a time, there was a mysterious island where strange creatures lived. Continue the story..."
]

# 检查调试模式（现在通过命令行参数设置）
if os.getenv('DEBUG', '0').lower() in ('1', 'true', 'yes'):
    import debugpy
    try:
        # 使用异常处理适配多进程代码，这样只有一个进程会监听5678端口
        debugpy.listen(("localhost", 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print("Debugger attached")
    except Exception as e:
        # 如果端口已被占用，忽略异常（可能是其他进程已启动调试）
        pass
for i, prompt in enumerate(prompts):
    # 编码输入文本
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,    # 启用填充以确保批处理兼容性
        max_length=512   # 设置最大输入长度
    )
    
    # 将输入张量移动到模型所在的设备
    inputs = inputs.to(model.device)
    
    # 计算当前输入的序列长度，用于 cache_position
    input_length = inputs.input_ids.shape[1]



    # 使用 model.generate() 进行文本生成
    with torch.no_grad():  # 禁用梯度计算以节省内存
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # 控制新生成 token 的数量
            num_return_sequences=1,
            generation_config=generation_config,  # 传入生成配置
            pad_token_id=tokenizer.pad_token_id,  # 明确指定 pad_token_id
            use_cache = True
        )
    
    # 解码生成结果
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n{prompt}: {generated_text[len(prompt):]}")
