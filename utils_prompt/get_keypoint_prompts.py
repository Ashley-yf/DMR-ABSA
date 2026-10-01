import json
import sys, os

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from utils_prompt.qnaigc_api import QNAIGCDeepSeek
# AGS (Attribute-Grounded Synthesis) attribute candidate libraries
from attribute_candidates.objects import *
from attribute_candidates.categories import *
from attribute_candidates.aspects import *
from attribute_candidates.opinions import *
from attribute_candidates.sentiments import *
from prompts.keypoint_driven import *
from prompts.templates import *
from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())

def get_keypoint_prompts(domain, dataset, seed, num_shots, num_samples, num_examples):
    # 构建正确的数据文件路径
    if dataset == "sample":
        data_file = f'data/{domain}/{dataset}{num_shots}_all.json'
    else:
        data_file = f'data/{domain}/{dataset}_all.json'
    with open(data_file, 'r', encoding='utf-8-sig') as file:
        train = json.load(file)

    train_exp = []
    for t in train:
        if len(t['aspects']) > 0:
            train_exp.append(t)
    print(f"Number of examples with aspects: {len(train_exp)}")

    sent_type = sentiments

    if "res" in domain:
        obj = res_objects
        ac = res_ac
        at = [at for v in res_at.values() for at in v] # all aspects
        ot = res_opinions
    elif domain == "lap":
        obj = lap_objects
        ac = lap_ac
        at = [at for v in lap_at.values() for at in v] # all aspects
        ot = lap_opinions

    import random
    random.seed(seed)
    llm = QNAIGCDeepSeek()  # 替换为你的实际API key
    # 构建正确的输出文件名
    if dataset == "sample":
        output_file = f'prompts/keypoint_prompts/{domain}_{dataset}{num_shots}.json'
    else:
        output_file = f'prompts/keypoint_prompts/{domain}_{dataset}.json'

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 清空输出文件（如果存在）
    with open(output_file, 'w', encoding='utf-8-sig') as file:
        file.write("")  # 清空文件

    print(f"开始处理 {num_samples} 个样本，结果将增量写入: {output_file}")

    output = []
    for i in range(num_samples):
        prompt = {}
        prompt['ID'] = i
        prompt['object'] = random.choice(obj)
        prompt['aspect'] = random.choice(at)
        prompt['category'] = random.choice(ac)
        ot_for_cat = ot[prompt['category']]
        prompt['opinion'] = random.choice(ot_for_cat)
        prompt['sentiment'] = random.choice(sent_type)
        prompt['examples'] = []
        for _ in range(num_examples):
            prompt['examples'].append(random.choice(train_exp))
        opinion = prompt['opinion'][0]
        if prompt['sentiment'] == "a consistent sentiment":
            sentiment = prompt['sentiment'] + f" ({prompt['opinion'][1]})"
        else:
            sentiment = prompt['sentiment']
        keypoint_prompt = generate_prompt.replace("{domain}", "restaurant" if "res" in domain else "laptop")\
            .replace("{object}", prompt['object'])\
            .replace("{aspect}", prompt['aspect']).replace("{category}", prompt['category'])\
            .replace("{opinion}", opinion).replace("{sentiment}", sentiment)
        # 构建完整的keypoint_prompt，包含所有examples
        for example in prompt['examples']:
            keypoint_prompt += get_input_template(example)
            label = [[asp["target"], asp["polarity"]] for asp in example['aspects']]
            keypoint_prompt += f"{label}\n\n"

        #print(f"=== 样本 {i+1}/{num_samples} 的完整keypoint_prompt ===")
        print(keypoint_prompt)

        # 在构建完完整prompt后，调用一次大模型
        gen_result = llm.generate_sentence_from_prompt(
            keypoint_prompt=keypoint_prompt
        )
        #print("LLM返回结果:", gen_result)
        if gen_result["success"]:
            sentences = gen_result["sentence"] + "\n" + gen_result["label"]
            #print("解析后的句子和标签:", sentences)
        else:
            # 如果调用失败，用空串占位，便于后续排查
            sentences = ""
            #print("错误 LLM调用失败，返回内容：", gen_result.get("error", "未知错误"))
        
        prompt['prompt_gen'] = keypoint_prompt + "llm_output:" + sentences
        #print(prompt['prompt_gen'])

        # 增量写入：将每条记录立即写入文件
        with open(output_file, 'a', encoding='utf-8-sig') as file:
            if i == 0:
                # 第一条记录：写入开头的 [ 和记录
                file.write("[\n")
                file.write(json.dumps(prompt, indent=4, ensure_ascii=False))
            else:
                # 后续记录：写入逗号和记录
                file.write(",\n")
                file.write(json.dumps(prompt, indent=4, ensure_ascii=False))

        print(f"[OK] 样本 {i+1}/{num_samples} 已写入文件: {output_file}")

    # 处理完所有样本后，关闭JSON数组
    with open(output_file, 'a', encoding='utf-8-sig') as file:
        file.write("\n]")

    print(f"[完成] 所有样本处理完成！结果已保存到: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate keypoint prompts')
    parser.add_argument('--domain', type=str, required=True, help='Domain (res, lap, etc.)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_shots', type=int, default=2, help='Number of shots')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--num_examples', type=int, default=4, help='Number of examples per prompt')

    args = parser.parse_args()

    get_keypoint_prompts(
        domain=args.domain,
        dataset=args.dataset,
        seed=args.seed,
        num_shots=args.num_shots,
        num_samples=args.num_samples,
        num_examples=args.num_examples
    )

