import json
import sys, os
import spacy
import random

from tqdm import tqdm

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

spacy.require_cpu()
from utils_prompt.qnaigc_api import QNAIGCDeepSeek
import random
from prompts.instance_driven import *
from prompts.templates import *
from dotenv import find_dotenv,load_dotenv
_=load_dotenv(find_dotenv())

nlp = spacy.load("en_core_web_sm")


def mask_aspect_terms(sentence, aspects, window_size=2):
    doc = nlp(sentence)
    masks = [False] * len(doc)
    for aspect in aspects:
        start_token_idx = None
        end_token_idx = None
        for token in doc:
            if token.idx >= int(aspect['from']):
                start_token_idx = token.i
                break
        for token in doc:
            if token.idx + len(token.text) >= int(aspect['to']):
                end_token_idx = token.i
                break
        # print(sentence)
        assert start_token_idx is not None and end_token_idx is not None
        start_idx = max(start_token_idx - window_size, 0)
        end_idx = min(end_token_idx + window_size, len(doc) - 1)

        for i in range(start_idx, end_idx + 1):
            masks[i] = True

    masked_sentence = []
    for i, token in enumerate(doc):
        if masks[i]:
            masked_sentence.append("<mask>")
        else:
            masked_sentence.append(token.text)

    final_sentence = []
    previous_mask = False
    for word in masked_sentence:
        if word == "<mask>" and previous_mask:
            continue
        final_sentence.append(word)
        previous_mask = (word == "<mask>")

    return ' '.join(final_sentence)


def mask_context(sentence, aspects, sample_times=2):
    doc = nlp(sentence)
    masks = [False] * len(doc)

    for aspect in aspects:
        start = int(aspect['from'])
        end = int(aspect['to'])
        for token in doc:
            if not (token.idx >= end or token.idx + len(token.text) <= start):
                masks[token.i] = True
    outputs = []
    for i in range(sample_times):
        max_start = int(len(doc) * 0.4)
        start_point = random.randint(0, max_start)
        end_point = start_point + int(len(doc) * 0.6)
        end_point = min(end_point, len(doc))

        masked_sentence = []
        for i, token in enumerate(doc):
            if start_point <= i < end_point and not masks[i]:
                masked_sentence.append("<mask>")
            else:
                masked_sentence.append(token.text)

        final_sentence = []
        previous_mask = False
        for word in masked_sentence:
            if word == "<mask>" and previous_mask:
                continue
            final_sentence.append(word)
            previous_mask = (word == "<mask>")
        outputs.append(' '.join(final_sentence))
    return outputs


def get_instance_prompts(domain, dataset, seed, num_shots):
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

    random.seed(seed)
    llm = QNAIGCDeepSeek()

    # 构建正确的输出文件名
    if dataset == "sample":
        output_file = f'prompts/instance_prompts/{domain}_{dataset}{num_shots}.json'
    else:
        output_file = f'prompts/instance_prompts/{domain}_{dataset}.json'

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 清空输出文件（如果存在）
    with open(output_file, 'w', encoding='utf-8-sig') as file:
        file.write("")  # 清空文件

    print(f"开始处理 instance prompts，结果将增量写入: {output_file}")

    # 跟踪是否是第一次写入
    first_write = True

    def write_incremental(prompt_data, is_first):
        """增量写入单个prompt到JSON文件"""
        nonlocal first_write

        with open(output_file, 'a', encoding='utf-8-sig') as file:
            if first_write:
                # 第一次写入：开始JSON数组
                file.write("[\n")
                file.write(json.dumps(prompt_data, indent=4, ensure_ascii=False))
                first_write = False
            else:
                # 后续写入：添加逗号分隔符
                file.write(",\n")
                file.write(json.dumps(prompt_data, indent=4, ensure_ascii=False))

    # 采样20k个样本
    output = []
    type = "paraphrase_one"
    batch_bar = tqdm(total=len(train_exp), initial=0, desc="paraphrase_one进度")
    for i in range(len(train_exp)):
        prompt = {}
        prompt['ID'] = train_exp[i]['ID'] + f"_{type}"
        example = train_exp[i]
        example_prompt = get_input_template(example)
        label = [[asp["target"], asp["polarity"]] for asp in example['aspects']]
        example_prompt += f"{label}\n\n"
        prompt_aug = paraphrase_prompt.replace("{domain}", "restaurant" if "res" in domain else "laptop")
        prompt_aug += example_prompt
        # print(prompt_aug)
        result1 = llm.generate_sentence_from_prompt(
            keypoint_prompt=prompt_aug,
        )
        if result1["success"]:
            sentences1 = result1["sentence"] + "\n" + result1["label"]
        else:
            sentences1 = "LLM调用失败"
        prompt['prompt_aug'] = prompt_aug + "4 Diverse Paraphrased Sentences with Labels:\n\n1. Sentence: " + sentences1

        # 增量写入
        write_incremental(prompt, True)
        batch_bar.update(1)

        if batch_bar.n % 10 == 0:  # 每处理10条记录打印一次进度
            print(f"[OK] 已处理 {batch_bar.n}/{batch_bar.total} 条记录")

    type = "paraphrase_comb"
    combined_set = set()

    #MAX_COMBINE = 1000
    MAX_COMBINE = 1000

    # Main loop to generate combinations
    batch_bar = tqdm(total=MAX_COMBINE, initial=len(combined_set), desc="paraphrase_comb进度")
    while len(combined_set) < MAX_COMBINE:
        i, j = random.randint(0, len(train_exp) - 1), random.randint(0, len(train_exp) - 1)
        if i != j:  # Ensure different samples
            combo_id = f"{train_exp[i]['ID']}_{train_exp[j]['ID']}_{type}"

            # Check if this combination has already been used
            if combo_id not in combined_set:
                combined_set.add(combo_id)  # Mark this combination as used
                prompt = {}
                prompt['ID'] = combo_id
                example1 = train_exp[i]
                example2 = train_exp[j]

                example1_prompt = get_input_template(example1)
                label1 = [[asp["target"], asp["polarity"]] for asp in example1['aspects']]
                example1_prompt += f"{label1}\n\n"
                example2_prompt = get_input_template(example2)
                label2 = [[asp["target"], asp["polarity"]] for asp in example2['aspects']]
                example2_prompt += f"{label2}\n\n"

                prompt_aug = combination_prompt.replace("{domain}", "restaurant" if "res" in domain else "laptop")
                prompt_aug += "1. " + example1_prompt + "2. " + example2_prompt
                result2 = llm.generate_sentence_from_prompt(
                    keypoint_prompt=prompt_aug
                )
                if result2["success"]:
                    sentences2 = result2["sentence"] + "\n" + result2["label"]
                else:
                    sentences2 = "LLM调用失败"
                prompt[
                    'prompt_aug'] = prompt_aug + "4 Diverse Combined Sentences with Labels:\n\n1. Sentence: " + sentences2

                # 增量写入
                write_incremental(prompt, False)
        batch_bar.update(1)
        # Break if all possible combinations have been tried
        if len(combined_set) == len(train_exp) * (len(train_exp) - 1):
            break

    type = "aspect_mask"
    batch_bar = tqdm(total=len(train_exp), initial=0, desc="aspect_mask进度")
    for i in range(len(train_exp)):
        for window in [0, 2]:
            prompt = {}
            prompt['ID'] = train_exp[i]['ID'] + f"_{type}_{window}"
            # mask aspect terms
            masked_sentence = mask_aspect_terms(train_exp[i]['sentence'], train_exp[i]['aspects'], window_size=window)
            example = train_exp[i]
            example_prompt = get_input_template(example)
            label = [[asp["target"], asp["polarity"]] for asp in example['aspects']]
            example_prompt += f"{label}\n\n"
            prompt_aug = mask_generate_prompt.replace("{domain}", "restaurant" if "res" in domain else "laptop") \
                .replace("{mask}", masked_sentence)
            prompt_aug += example_prompt
            result3 = llm.generate_sentence_from_prompt(
                keypoint_prompt=prompt_aug
            )
            if result3["success"]:
                sentences3 = result3["sentence"] + "\n" + result3["label"]
            else:
                sentences3 = "LLM调用失败"
            prompt['prompt_aug'] = prompt_aug + "4 Diverse Reconstructed Sentences with Labels:\n\n1. Sentence: " + sentences3

            # 增量写入
            write_incremental(prompt, False)
            batch_bar.update(1)

    type = "context_mask"
    batch_bar = tqdm(total=len(train_exp), initial=0, desc="context_mask进度")
    for i in range(len(train_exp)):
        # mask context words
        masked_sentences = mask_context(train_exp[i]['sentence'], train_exp[i]['aspects'], sample_times=2)
        for idx, masked_sentence in enumerate(masked_sentences):
            prompt = {}
            prompt['ID'] = train_exp[i]['ID'] + f"_{type}_{idx}"
            example = train_exp[i]
            example_prompt = get_input_template(example)
            label = [[asp["target"], asp["polarity"]] for asp in example['aspects']]
            example_prompt += f"{label}\n\n"
            prompt_aug = mask_generate_prompt.replace("{domain}", "restaurant" if "res" in domain else "laptop") \
                .replace("{mask}", masked_sentence)
            prompt_aug += example_prompt
            result4 = llm.generate_sentence_from_prompt(
                keypoint_prompt=prompt_aug,
            )
            if result4["success"]:
                sentences4 = result4["sentence"] + "\n" + result4["label"]
            else:
                sentences4 = "LLM调用失败"
            prompt[
                'prompt_aug'] = prompt_aug + "4 Diverse Reconstructed Sentences with Labels:\n\n1. Sentence: " + sentences4

            # 增量写入
            write_incremental(prompt, False)
            batch_bar.update(1)

    # 关闭JSON数组（只有在至少写入了一条记录的情况下）
    if not first_write:  # 如果有数据被写入
        with open(output_file, 'a', encoding='utf-8-sig') as file:
            file.write("\n]")

    print(f"[完成] 所有 instance prompts 处理完成！结果已保存到: {output_file}")

    return []  # 返回空列表，因为数据已经写入文件


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate instance prompts')
    parser.add_argument('--domain', type=str, required=True, help='Domain (res, lap, etc.)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_shots', type=int, default=2, help='Number of shots')

    args = parser.parse_args()

    get_instance_prompts(
        domain=args.domain,
        dataset=args.dataset,
        seed=args.seed,
        num_shots=args.num_shots
    )
