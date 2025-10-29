import json
from collections import Counter, defaultdict
import statistics
import pickle
import os
from transformers import AutoTokenizer
import pandas
import time

def is_valid_multiturn_pattern(conversations):
    """
    检查对话是否遵循 human->gpt->human->gpt... 的多轮对话模式
    
    Args:
        conversations: 对话列表
        
    Returns:
        bool: 是否符合多轮对话模式
    """
    if not conversations:
        return False
    
    for i, turn in enumerate(conversations):
        role = turn.get('from', '')
        
        # 检查角色交替模式
        if i % 2 == 0:  # 偶数位置应该是human
            if role != 'human':
                return False
        else:  # 奇数位置应该是gpt
            if role != 'gpt':
                return False
    
    return True

def analyze_conversation_dataset(data):
    """
    分析对话数据集，检查角色序列并统计长度分布
    
    Args:
        data: 包含对话数据的数据集
        
    Returns:
        dict: 包含分析结果的字典
    """
    results = {
        'total_samples': 0,
        'valid_multiturn_patterns': 0,  # 符合多轮对话模式的样本
        'invalid_multiturn_patterns': 0,  # 不符合多轮对话模式的样本
        'single_turn_conversations': 0,  # 单轮对话样本
        'role_sequence_patterns': Counter(),
        'length_distribution': defaultdict(int),
        'multiturn_length_stats': {},  # 多轮对话的长度统计
        'all_length_stats': {},  # 所有样本的长度统计
        'sample_details': []
    }
    
    all_lengths = []
    multiturn_lengths = []
    
    # 遍历数据集中的每个样本
    for sample_idx, sample in enumerate(data):
        results['total_samples'] += 1
        conversations = sample.get('conversations', [])
        
        # 检查是否符合多轮对话模式
        is_multiturn = is_valid_multiturn_pattern(conversations)
        turn_count = len(conversations)
        
        if turn_count == 0:
            # 空对话，跳过
            continue
        elif turn_count == 2:
            results['single_turn_conversations'] += 1
        elif is_multiturn and turn_count > 2:
            results['valid_multiturn_patterns'] += 1
        else:
            results['invalid_multiturn_patterns'] += 1
        
        # 记录角色序列模式
        role_sequence = [turn.get('from', '') for turn in conversations]
        sequence_pattern = '->'.join(role_sequence)
        results['role_sequence_patterns'][sequence_pattern] += 1
        
        # 计算样本长度（拼接所有内容）
        full_text = ''
        for turn in conversations:
            full_text += turn.get('value', '')
        
        text_length = len(full_text)
        all_lengths.append(text_length)
        
        if is_multiturn and turn_count > 1:
            multiturn_lengths.append(text_length)
        
        # 记录长度分布
        length_bucket = text_length // 10000  # 按100的倍数分组
        results['length_distribution'][length_bucket] += 1
        
        # 记录样本详情
        results['sample_details'].append({
            'sample_id': sample_idx + 1,
            'role_sequence': sequence_pattern,
            'is_multiturn': is_multiturn,
            'length': text_length,
            'turn_count': turn_count,
            'is_single_turn': (turn_count == 1)
        })
    
    # 计算所有样本的长度统计信息
    if all_lengths:
        results['all_length_stats'] = {
            'min': min(all_lengths),
            'max': max(all_lengths),
            'mean': statistics.mean(all_lengths),
            'median': statistics.median(all_lengths),
            'stdev': statistics.stdev(all_lengths) if len(all_lengths) > 1 else 0,
            'sample_count': len(all_lengths)
        }
    
    # 计算多轮对话样本的长度统计信息
    if multiturn_lengths:
        results['multiturn_length_stats'] = {
            'min': min(multiturn_lengths),
            'max': max(multiturn_lengths),
            'mean': statistics.mean(multiturn_lengths),
            'median': statistics.median(multiturn_lengths),
            'stdev': statistics.stdev(multiturn_lengths) if len(multiturn_lengths) > 1 else 0,
            'sample_count': len(multiturn_lengths)
        }
    
    return results

def print_analysis_results(results):
    """打印分析结果"""
    print("=" * 60)
    print("多轮对话数据集分析结果")
    print("=" * 60)
    
    print(f"总样本数: {results['total_samples']}")
    print(f"符合多轮对话模式的样本: {results['valid_multiturn_patterns']}")
    print(f"不符合多轮对话模式的样本: {results['invalid_multiturn_patterns']}")
    print(f"单轮对话样本: {results['single_turn_conversations']}")
    
    valid_ratio = (results['valid_multiturn_patterns'] / results['total_samples'] * 100) if results['total_samples'] > 0 else 0
    print(f"多轮对话模式符合率: {valid_ratio:.2f}%")
    
    print("\n角色序列模式统计 (前10种):")
    for pattern, count in results['role_sequence_patterns'].most_common(10):
        print(f"  {pattern}: {count}")
    
    print("\n所有样本长度统计:")
    stats = results['all_length_stats']
    print(f"  样本数: {stats.get('sample_count', 0)}")
    print(f"  最小值: {stats.get('min', 0)}")
    print(f"  最大值: {stats.get('max', 0)}")
    print(f"  平均值: {stats.get('mean', 0):.2f}")
    print(f"  中位数: {stats.get('median', 0)}")
    print(f"  标准差: {stats.get('stdev', 0):.2f}")
    
    if results['multiturn_length_stats']:
        print("\n多轮对话样本长度统计:")
        mt_stats = results['multiturn_length_stats']
        print(f"  样本数: {mt_stats.get('sample_count', 0)}")
        print(f"  最小值: {mt_stats.get('min', 0)}")
        print(f"  最大值: {mt_stats.get('max', 0)}")
        print(f"  平均值: {mt_stats.get('mean', 0):.2f}")
        print(f"  中位数: {mt_stats.get('median', 0)}")
        print(f"  标准差: {mt_stats.get('stdev', 0):.2f}")
    
    print("\n长度分布 (按100字符分组):")
    for length_bucket in sorted(results['length_distribution'].keys()):
        count = results['length_distribution'][length_bucket]
        percentage = count / results['total_samples'] * 100 if results['total_samples'] > 0 else 0
        print(f"  {length_bucket}-{length_bucket*10000+9999}: {count} ({percentage:.2f}%)")

# 打印角色分布。 
def get_multi_turn_distribution(sample_data):
    role_seqs = []
    for sample_idx, sample in enumerate(sample_data):
        messages = sample.get('messages', [])
        role_seq = []
        for message in messages:
            role_seq.append(message['role'])
            
        role_seqs.append('->'.join(role_seq))
    
    count_res = Counter(role_seqs)
    print(count_res)

# 获取tokenizer之后的分布
def get_distribute(sample_data):
    chat_count = [0] * 20 
    sample_lens = []
    for sample_idx, sample in enumerate(sample_data[0:3000]):
        messages = sample.get('messages', [])
        # messages 默认就是deepseek的格式
        all_ids = tokenizer.apply_chat_template(messages)
        sample_lens.append(len(all_ids))

        # 每个包含1轮对话位于0号位， 19轮对话保存超过>=20轮以上的内容
        chat_turn = len(messages) // 2
        if  chat_turn < 20:
            chat_count[chat_turn-1] += 1
        else:
            chat_count[-1] += 1
    print(f"Total samples: {sum(chat_count)}, multi-turn: {chat_count}")


    pd = pandas.Series(sample_lens)  # 现在samples包含的是token长度
    print("Token length distribution:")
    print(pd.describe())
    

# 示例使用
if __name__ == "__main__":
    # /mnt/self-define/dongjie/model/dataset/tulu3/data/tulu-3-sft-mixture/tulu_v3_mix.jsonl  董杰
    # /mnt/self-define/sunning/tulu-3-sft-mixture.jsonl

    data_file="/mnt/self-define/sunning/tulu-3-sft-mixture.jsonl"
    tulu3_file="openr1_full.jsonl"
    # r1_file= "r1_2000.jsonl"
    tulu3_pkl = "tulu3-sn.pkl"
    tokenizer = AutoTokenizer.from_pretrained("/public/model/Meta-Llama-3.1-8B-Instruct/")
    if not os.path.exists(tulu3_pkl):
        sample_data = []
        with open(data_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                data = json.loads(line)
                sample_data.append(data)
        with open(tulu3_pkl, "wb") as f:
            print(f"数据已保存到 {tulu3_pkl}")
            pickle.dump(sample_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(tulu3_pkl, "rb") as f:
            print(f"读取序列化文件{tulu3_pkl}")
            start = time.time()
            sample_data = pickle.load(f)
            end = time.time()  # 记录结束时间
            print(f"耗时：{end - start:.4f} 秒")
    


    # # 分析数据
    # analysis_results = analyze_conversation_dataset(sample_data)
    # # 打印结果
    # print_analysis_results(analysis_results)
    get_multi_turn_distribution(sample_data)
    get_distribute(sample_data)
            
    
    
    # # 可选：保存详细结果到文件
    # with open('multiturn_conversation_analysis.json', 'w', encoding='utf-8') as f:
    #     json.dump(analysis_results, f, indent=2, ensure_ascii=False)