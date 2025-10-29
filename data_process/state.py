import json

result=[]

def analyze_jsonl(file_path):
    valid_sample_count=0
    total_samples = 0
    total_user_messages = 0
    total_assistant_messages = 0
    
    # 存储不符合标准的样本
    multi_turn_samples = []  # 多轮对话样本
    invalid_order_samples = []  # 顺序错误的样本
    single_role_samples = []  # 只有一种角色的样本

    multi_turn_instances = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            total_samples += 1
            try:
                data = json.loads(line)
                sample_id = data.get('id', f"line_{line_num}")
                messages = data.get('messages', [])
                
                user_count = 0
                assistant_count = 0
                roles = []
                
                # 分析每个消息
                for message in messages:
                    role = message.get('role', '').lower()
                    roles.append(role)
                    
                    if role == 'user':
                        user_count += 1
                        total_user_messages += 1
                    elif role == 'assistant':
                        assistant_count += 1
                        total_assistant_messages += 1
                
                # 检查样本是否符合标准
                if user_count > 1 or assistant_count > 1:
                    multi_turn_samples.append({
                        'id': sample_id,
                        'user_count': user_count,
                        'assistant_count': assistant_count,
                        'roles': roles
                    })
                
                # 检查顺序是否正确
                if len(roles) > 0:
                    # 检查是否以user开始
                    if roles[0] != 'user':
                        invalid_order_samples.append({
                            'id': sample_id,
                            'reason': f"以'{roles[0]}'开始而不是'user'",
                            'roles': roles
                        })
                    
                    # 检查是否有连续相同角色
                    for i in range(1, len(roles)):
                        if roles[i] == roles[i-1]:
                            invalid_order_samples.append({
                                'id': sample_id,
                                'reason': f"位置{i-1}和{i}有连续的'{roles[i]}'角色",
                                'roles': roles
                            })
                            break
                
                # 检查是否只有一种角色
                if user_count > 0 and assistant_count == 0:
                    single_role_samples.append({
                        'id': sample_id,
                        'reason': "只有user消息",
                        'roles': roles
                    })
                elif assistant_count > 0 and user_count == 0:
                    single_role_samples.append({
                        'id': sample_id,
                        'reason': "只有assistant消息",
                        'roles': roles
                    })

                if (user_count == assistant_count) and assistant_count > 1:
                    multi_turn_instances.append(data)
                    valid_sample_count +=1
                
                # if len(roles) ==2 and roles[0]=='user' and roles[1]=='assistant':
                    
                #     result.append(data)
                #     valid_sample_count +=1
                # if len(roles) ==3 and roles[0] == 'system' and roles[1]=='user' and roles[2]=='assistant':
                #     result.append(data)
                #     valid_sample_count +=1

        
            except json.JSONDecodeError:
                print(f"警告: 第 {line_num} 行无法解析为JSON")

    # print(f"有效样本{valid_sample_count}/{total_samples}")       
    # with open("new_standard_tulu3.json", 'w') as f:
    #     for item in result:
    #         json_line = json.dumps(item, ensure_ascii=False)
    #         f.write(json_line + '\n')

    
    return {
        'total_samples': total_samples,
        'total_user_messages': total_user_messages,
        'total_assistant_messages': total_assistant_messages,
        'multi_turn_samples': multi_turn_samples,
        'invalid_order_samples': invalid_order_samples,
        'single_role_samples': single_role_samples
    }

def print_results(results):
    print(f"总样本数: {results['total_samples']}")
    print(f"用户消息数: {results['total_user_messages']}")
    print(f"助手消息数: {results['total_assistant_messages']}")
    print(f"总对话数: {results['total_user_messages'] + results['total_assistant_messages']}")
    
    if results['total_samples'] > 0:
        avg_messages = (results['total_user_messages'] + results['total_assistant_messages']) / results['total_samples']
        print(f"平均每样本对话数: {avg_messages:.2f}")
    
    # 打印多轮对话样本
    if results['multi_turn_samples']:
        print("\n多轮对话样本:")
        for sample in results['multi_turn_samples']:
            print(f"ID: {sample['id']}, User消息数: {sample['user_count']}, Assistant消息数: {sample['assistant_count']}")
            print(f"  角色顺序: {' -> '.join(sample['roles'])}")
    
    # 打印顺序错误样本
    if results['invalid_order_samples']:
        print("\n顺序错误样本:")
        for sample in results['invalid_order_samples']:
            print(f"ID: {sample['id']}, 原因: {sample['reason']}")
            print(f"  角色顺序: {' -> '.join(sample['roles'])}")
    
    # 打印单一角色样本
    if results['single_role_samples']:
        print("\n单一角色样本:")
        for sample in results['single_role_samples']:
            print(f"ID: {sample['id']}, 原因: {sample['reason']}")
            print(f"  角色顺序: {' -> '.join(sample['roles'])}")

if __name__ == "__main__":
    file_path = "/mnt/self-define/dongjie/model/dataset/tulu3/data/tulu-3-sft-mixture/tulu_v3_mix.jsonl"  # 替换为你的JSONL文件路径
    # file_path = "/mnt/self-define/songquanheng/dataset/tulu3/data/tulu-3-sft-mixture/tulu_v3_mix_v3.json"
    #file_path = "/mnt/self-define/songquanheng/toolkits/new_standard_tulu3.json"
    results = analyze_jsonl(file_path)
    print_results(results)
    
    # 打印统计摘要
    print("\n统计摘要:")
    print(f"多轮对话样本数: {len(results['multi_turn_samples'])}")
    print(f"顺序错误样本数: {len(results['invalid_order_samples'])}")
    print(f"单一角色样本数: {len(results['single_role_samples'])}")
    print(f"标准样本比例: {(results['total_samples'] - len(results['multi_turn_samples']) - len(results['invalid_order_samples']) - len(results['single_role_samples'])) / results['total_samples'] * 100:.2f}%")
