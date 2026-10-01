#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方面词集合匹配示例说明
详细展示"方面词集合完全匹配"的含义和判断逻辑
"""

def extract_aspect_set(label):
    """从Label字段提取方面词集合"""
    aspect_set = set()
    try:
        for item in label:
            if len(item) >= 1:
                aspect_set.add(item[0])  # 只取方面词，忽略情感极性
    except Exception as e:
        print(f"提取方面词时出错: {e}")

    return aspect_set

def explain_aspect_matching():
    """解释方面词集合匹配的概念"""

    print("=" * 60)
    print("方面词集合完全匹配 - 详细说明")
    print("=" * 60)

    # 示例数据
    examples = [
        {
            "name": "示例1：完全匹配",
            "data1": {
                "Sentence": "The food quality was excellent and service was poor.",
                "Label": [
                    ["food quality", "positive"],
                    ["service", "negative"]
                ]
            },
            "data2": {
                "Sentence": "Excellent food quality but terrible service experience.",
                "Label": [
                    ["service", "negative"],
                    ["food quality", "positive"]
                ]
            }
        },
        {
            "name": "示例2：部分匹配（不完全匹配）",
            "data1": {
                "Sentence": "The food quality was excellent.",
                "Label": [
                    ["food quality", "positive"]
                ]
            },
            "data2": {
                "Sentence": "The food quality was excellent and service was poor.",
                "Label": [
                    ["food quality", "positive"],
                    ["service", "negative"]
                ]
            }
        },
        {
            "name": "示例3：完全不匹配",
            "data1": {
                "Sentence": "The food quality was excellent.",
                "Label": [
                    ["food quality", "positive"]
                ]
            },
            "data2": {
                "Sentence": "The service attitude was friendly.",
                "Label": [
                    ["service attitude", "positive"]
                ]
            }
        },
        {
            "name": "示例4：重复方面词（去重后匹配）",
            "data1": {
                "Sentence": "Food quality good, food quality excellent, service poor.",
                "Label": [
                    ["food quality", "positive"],
                    ["food quality", "positive"],  # 重复的方面词
                    ["service", "negative"]
                ]
            },
            "data2": {
                "Sentence": "Excellent food quality and terrible service.",
                "Label": [
                    ["food quality", "positive"],
                    ["service", "negative"]
                ]
            }
        },
        {
            "name": "示例5：情感极性不同但方面词相同",
            "data1": {
                "Sentence": "The food quality was excellent.",
                "Label": [
                    ["food quality", "positive"]
                ]
            },
            "data2": {
                "Sentence": "The food quality was terrible.",
                "Label": [
                    ["food quality", "negative"]  # 情感极性不同
                ]
            }
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{example['name']}")
        print("-" * 40)

        # 提取方面词集合
        set1 = extract_aspect_set(example["data1"]["Label"])
        set2 = extract_aspect_set(example["data2"]["Label"])

        print(f"数据1的方面词集合: {set1}")
        print(f"数据2的方面词集合: {set2}")
        print(f"集合是否相等: {set1 == set2}")

        # 显示原始Label
        print(f"数据1的原始Label: {example['data1']['Label']}")
        print(f"数据2的原始Label: {example['data2']['Label']}")

        print(f"句子1: {example['data1']['Sentence']}")
        print(f"句子2: {example['data2']['Sentence']}")

        # 判断是否为方面词集合匹配
        if set1 == set2:
            print("[MATCH] 方面词集合完全匹配")
            print("        -> 进入下一步：句子相似度判断")
        else:
            print("[NO MATCH] 方面词集合不完全匹配")
            print("          -> 直接判定为不重复，无需计算句子相似度")

    print("\n" + "=" * 60)
    print("核心要点总结")
    print("=" * 60)
    print("1. 集合匹配：只比较方面词，忽略情感极性")
    print("2. 顺序无关：方面词的排列顺序不影响匹配")
    print("3. 自动去重：相同的方面词在集合中只保留一个")
    print("4. 必须完全相等：两个集合必须包含完全相同的方面词")
    print("5. 只有集合匹配才会进行句子相似度计算")

if __name__ == "__main__":
    explain_aspect_matching()