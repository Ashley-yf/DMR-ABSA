#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将处理后文件（property-driven/res/direct_res_sample5.json）中正确的Label更新到原始keypoint提示词文件（res_sample5.json）的prompt_gen中
专门用于修复原始数据中的Label部分
"""

import json
import os
import sys

# 仓库根目录（scripts/ 的上一级）
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fix_labels_in_original_file():
    """
    使用处理后的正确Label更新原始文件中的prompt_gen
    """
    # 文件路径（可用命令行参数覆盖: python scripts/fix_labels_from_processed.py [processed_file] [original_file]）
    processed_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        REPO_ROOT, "property-driven", "res", "direct_res_sample5.json")
    original_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        REPO_ROOT, "prompts", "keypoint_prompts", "res_sample5.json")

    print("=== Label修复工具 ===")
    print(f"处理后文件: {processed_file}")
    print(f"原始文件: {original_file}")
    print()

    # 检查文件是否存在
    if not os.path.exists(processed_file):
        print(f"错误: 处理后的文件不存在 - {processed_file}")
        return False

    if not os.path.exists(original_file):
        print(f"错误: 原始文件不存在 - {original_file}")
        return False

    try:
        # 第一步：读取处理后的文件，提取正确的Label
        print("正在读取处理后的文件...")
        with open(processed_file, 'r', encoding='utf-8-sig') as f:
            processed_data = json.load(f)

        # 创建ID到Label的映射
        id_to_labels = {}
        for item in processed_data:
            item_id = str(item.get("ID", ""))
            labels = item.get("label", [])
            if item_id and labels:
                id_to_labels[item_id] = labels

        print(f"提取了 {len(id_to_labels)} 个ID的正确Label信息")

        # 第二步：读取原始文件
        print("\n正在读取原始文件...")
        with open(original_file, 'r', encoding='utf-8-sig') as f:
            original_data = json.load(f)

        # 第三步：更新原始文件中的Label
        print("开始更新原始文件中的Label...")
        updated_count = 0
        skipped_count = 0

        for item in original_data:
            try:
                item_id = str(item.get("ID", ""))
                prompt_gen = item.get("prompt_gen", "")

                # 检查条件
                if item_id not in id_to_labels:
                    skipped_count += 1
                    continue

                if "llm_output:" not in prompt_gen:
                    print(f"ID {item_id}: 跳过（无llm_output）")
                    skipped_count += 1
                    continue

                if "Label:" not in prompt_gen:
                    print(f"ID {item_id}: 跳过（无Label）")
                    skipped_count += 1
                    continue

                # 获取正确的Label
                correct_labels = id_to_labels[item_id]

                # 分割prompt_gen
                parts = prompt_gen.split("llm_output:", 1)
                if len(parts) != 2:
                    print(f"ID {item_id}: 跳过（llm_output格式异常）")
                    skipped_count += 1
                    continue

                before_llm_output = parts[0]
                llm_output_part = parts[1]

                # 分离llm_output的句子和Label部分
                if "Label:" in llm_output_part:
                    sentence_part = llm_output_part.split("Label:")[0].strip()
                    original_label_part = llm_output_part.split("Label:")[1].strip()
                else:
                    print(f"ID {item_id}: 跳过（无Label部分）")
                    skipped_count += 1
                    continue

                # 构建新的Label部分
                new_label_part = str(correct_labels)

                # 重新组合prompt_gen
                new_prompt_gen = before_llm_output + "llm_output:" + sentence_part + "\nLabel: " + new_label_part

                # 更新数据
                item["prompt_gen"] = new_prompt_gen
                updated_count += 1

                # 显示更新示例（前3个）
                if updated_count <= 3:
                    print(f"ID {item_id}: [OK] Label已更新")
                    print(f"  原Label: {original_label_part[:80]}...")
                    print(f"  新Label: {new_label_part}")

            except Exception as e:
                print(f"ID {item.get('ID', 'unknown')}: [ERROR] 更新失败 - {e}")
                skipped_count += 1

        # 第四步：保存文件
        print(f"\n保存结果:")
        print(f"  成功更新: {updated_count} 条记录")
        print(f"  跳过记录: {skipped_count} 条记录")

        # 创建备份
        backup_path = original_file.replace('.json', '_before_label_fix.json')
        with open(backup_path, 'w', encoding='utf-8-sig') as f:
            json.dump(original_data, f, ensure_ascii=False, indent=2)
        print(f"  备份文件: {backup_path}")

        # 保存更新后的原始文件
        with open(original_file, 'w', encoding='utf-8-sig') as f:
            json.dump(original_data, f, ensure_ascii=False, indent=2)
        print(f"  更新文件: {original_file}")

        return True

    except Exception as e:
        print(f"处理过程中出错: {e}")
        return False


def main():
    """
    主函数
    """
    print("Label修复工具")
    print("功能: 将处理后文件中的正确Label更新到原始keypoint提示词文件")
    print("=" * 60)

    success = fix_labels_in_original_file()

    if success:
        print("\n" + "=" * 60)
        print("[OK] Label修复完成!")
        print("=" * 60)
        print("\n建议:")
        print("1. 检查更新后的文件是否正确")
        print("2. 如果有问题，可以从备份文件恢复")
        print("3. 重新运行处理脚本验证结果")
        print("4. 删除此脚本文件（如不再需要）")
    else:
        print("\n[ERROR] Label修复失败，请检查错误信息")


if __name__ == "__main__":
    main()