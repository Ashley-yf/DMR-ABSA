#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理所有数据集的脚本
- 运行get_keypoint_prompts.py生成所有提示词文件
- 运行process_absa_data.py处理所有文件
"""

import os
import subprocess
import sys
import time
from typing import List, Tuple

# 保证从任意工作目录运行都能找到仓库根目录下的 utils_prompt 等包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_prompt.qnaigc_api import QNAIGCDeepSeek
from utils_prompt.get_keypoint_prompts import get_keypoint_prompts
from utils_prompt.get_instance_prompts import get_instance_prompts

# 仓库根目录（scripts/ 的上一级），所有相对路径以此为基准
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 定义域和数据集配置
DOMAINS = ["res", "lap", "res15", "res16"]
DATASETS_CONFIG = [
    ("sample", 2),
    ("sample", 5),
    ("test", None),
    ("dev", None),
    ("train", None)
]

# 配置参数
SEED = 42
NUM_SAMPLES = 4000
NUM_EXAMPLES = 4

def run_command(cmd: str, description: str) -> bool:
    """运行命令并检查结果"""
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=REPO_ROOT)

        if result.returncode == 0:
            print(f"成功 成功: {description}")
            return True
        else:
            print(f"错误 失败: {description}")
            print(f"错误输出: {result.stderr}")
            return False

    except Exception as e:
        print(f"错误 异常: {description} - {e}")
        return False

def generate_all_prompts() -> List[Tuple[str, str, int, str]]:
    """生成所有提示词文件"""
    print("启动 开始生成所有提示词文件...")

    generated_files = []
    failed_files = []

    for domain in DOMAINS:
        for dataset, num_shots in DATASETS_CONFIG:
            # 生成 keypoint prompts
            if num_shots:
                cmd1 = f'python -c "from utils_prompt.get_keypoint_prompts import get_keypoint_prompts; get_keypoint_prompts(\'{domain}\', \'{dataset}\', {SEED}, {num_shots}, {NUM_SAMPLES}, {NUM_EXAMPLES})"'
                description1 = f"keypoint_{domain}_{dataset}{num_shots}.json"
            else:
                # 对于test/dev/train，num_shots参数设为0（不会被使用）
                cmd1 = f'python -c "from utils_prompt.get_keypoint_prompts import get_keypoint_prompts; get_keypoint_prompts(\'{domain}\', \'{dataset}\', {SEED}, 0, {NUM_SAMPLES}, {NUM_EXAMPLES})"'
                description1 = f"keypoint_{domain}_{dataset}.json"

            success1 = run_command(cmd1, f"生成keypoint提示词: {description1}")

            if success1:
                generated_files.append((domain, dataset, num_shots or 0, "keypoint"))
                # 输出文件路径
                if dataset == "sample":
                    output_path1 = os.path.join(REPO_ROOT, "prompts", "keypoint_prompts", f"{domain}_{dataset}{num_shots}.json")
                else:
                    output_path1 = os.path.join(REPO_ROOT, "prompts", "keypoint_prompts", f"{domain}_{dataset}.json")

                # 检查文件是否存在
                if os.path.exists(output_path1):
                    file_size = os.path.getsize(output_path1)
                    print(f"   文件大小: {file_size:,} bytes")
                else:
                    print(f"   警告: 输出文件不存在 - {output_path1}")
            else:
                failed_files.append((domain, dataset, num_shots or 0, "keypoint"))

            # 添加延迟避免过快
            time.sleep(1)

            # 生成 instance prompts
            if num_shots:
                cmd2 = f'python -c "from utils_prompt.get_instance_prompts import get_instance_prompts; get_instance_prompts(\'{domain}\', \'{dataset}\', {SEED}, {num_shots})"'
                description2 = f"instance_{domain}_{dataset}{num_shots}.json"
            else:
                # 对于test/dev/train，num_shots参数设为0（不会被使用）
                cmd2 = f'python -c "from utils_prompt.get_instance_prompts import get_instance_prompts; get_instance_prompts(\'{domain}\', \'{dataset}\', {SEED}, 0)"'
                description2 = f"instance_{domain}_{dataset}.json"

            success2 = run_command(cmd2, f"生成instance提示词: {description2}")

            if success2:
                generated_files.append((domain, dataset, num_shots or 0, "instance"))
                # 输出文件路径
                if dataset == "sample":
                    output_path2 = os.path.join(REPO_ROOT, "prompts", "instance_prompts", f"{domain}_{dataset}{num_shots}.json")
                else:
                    output_path2 = os.path.join(REPO_ROOT, "prompts", "instance_prompts", f"{domain}_{dataset}.json")

                # 检查文件是否存在
                if os.path.exists(output_path2):
                    file_size = os.path.getsize(output_path2)
                    print(f"   文件大小: {file_size:,} bytes")
                else:
                    print(f"   警告: 输出文件不存在 - {output_path2}")
            else:
                failed_files.append((domain, dataset, num_shots or 0, "instance"))

            # 添加延迟避免过快
            time.sleep(1)

    print(f"\n统计 提示词生成完成:")
    print(f"   成功: {len(generated_files)} 个文件")
    print(f"   失败: {len(failed_files)} 个文件")

    if failed_files:
        print(f"\n错误 失败的文件:")
        for domain, dataset, num_shots, prompt_type in failed_files:
            print(f"   {prompt_type}_{domain}_{dataset}{num_shots if num_shots else ''}")

    return generated_files

def process_all_files():
    """处理所有生成的文件"""
    print(f"\n刷新 开始处理所有生成的文件...")

    # 定义需要处理的prompt目录和对应的输出目录
    prompt_configs = [
        {
            "input_dir": os.path.join(REPO_ROOT, "prompts", "keypoint_prompts"),
            "output_dir": os.path.join(REPO_ROOT, "property-driven"),
            "prompt_type": "keypoint"
        },
        {
            "input_dir": os.path.join(REPO_ROOT, "prompts", "instance_prompts"),
            "output_dir": os.path.join(REPO_ROOT, "seed-driven"),
            "prompt_type": "instance"
        }
    ]

    all_prompt_files = []

    # 收集所有目录中的JSON文件
    for config in prompt_configs:
        prompt_dir = config["input_dir"]
        prompt_type = config["prompt_type"]

        if not os.path.exists(prompt_dir):
            print(f"错误 目录不存在: {prompt_dir}")
            continue

        # 查找所有 JSON 文件
        prompt_files = []
        for file in os.listdir(prompt_dir):
            if file.endswith(".json"):
                prompt_files.append((file, prompt_dir, config["output_dir"], prompt_type))

        all_prompt_files.extend(prompt_files)
        print(f"在 {os.path.basename(prompt_dir)} 目录找到 {len(prompt_files)} 个 JSON 文件")

    print(f"\n总共找到 {len(all_prompt_files)} 个 JSON 文件:")
    for file, input_dir, output_dir, prompt_type in sorted(all_prompt_files):
        input_dir_name = os.path.basename(input_dir)
        output_dir_name = os.path.basename(output_dir)
        print(f"   {input_dir_name}/{file} -> {output_dir_name}")

    if not all_prompt_files:
        print("错误 未找到任何 JSON 文件")
        return False

    # 处理每个文件
    processed_count = 0
    failed_count = 0

    for file, input_dir, output_dir, prompt_type in all_prompt_files:
        domain = file.split("_")[0]
        input_file = os.path.join(input_dir, file)

        print(f"\n处理文件: {file} (类型: {prompt_type})")
        print(f"输入目录: {os.path.basename(input_dir)}")
        print(f"输出目录: {os.path.basename(output_dir)}")

        # 构建完整的命令参数，使用对应的输出目录
        cmd = f'python scripts/process_absa_data.py "{input_file}" "{output_dir}" "{file}" --prompt_type {prompt_type}'
        success = run_command(cmd, f"处理 {file}")

        if success:
            processed_count += 1
            print(f"   成功 处理成功: {file}")
        else:
            failed_count += 1
            print(f"   错误 处理失败: {file}")

        # 添加延迟避免过快处理 - 减少到1秒
        time.sleep(1)

    print(f"\n统计 文件处理完成:")
    print(f"   成功: {processed_count} 个文件")
    print(f"   失败: {failed_count} 个文件")
    print(f"   总计: {len(all_prompt_files)} 个文件")

    return processed_count > 0

def show_summary():
    """显示处理摘要"""
    print(f"\n" + "="*80)
    print("目标 批量处理摘要")
    print("="*80)
    print(f"处理配置:")
    print(f"  - 域: {', '.join(DOMAINS)}")
    print(f"  - 数据集: {len(DATASETS_CONFIG)} 种配置")
    print(f"  - 每个数据集生成 {NUM_SAMPLES} 个样本")
    print(f"  - 每个样本包含 {NUM_EXAMPLES} 个示例")
    print(f"  - 总预期输出: {len(DOMAINS) * len(DATASETS_CONFIG) * 2} 个文件 (keypoint + instance)")

    print(f"\n输出位置:")
    print(f"  - Keypoint提示词文件: prompts/keypoint_prompts/")
    print(f"  - Instance提示词文件: prompts/instance_prompts/")
    print(f"  - Keypoint处理结果: property-driven/")
    print(f"  - Instance处理结果: seed-driven/")

def main():
    """主函数"""
    print("启动 批量处理所有数据集")
    print("="*80)

    # 显示摘要
    show_summary()

    # 确认执行
    try:
        choice = input("\n是否继续执行? (y/n): ").lower().strip()
        if choice not in ['y', 'yes']:
            print("操作已取消")
            return
    except (EOFError, KeyboardInterrupt):
        print("\n操作已取消")
        return

    # 第一步：生成提示词
    generated_files = generate_all_prompts()

    if not generated_files:
        print("错误 提示词生成失败，无法继续处理")
        return

    # 第二步：处理文件
    success = process_all_files()

    if success:
        print(f"\n庆祝 批量处理完成！")
        print(f"所有文件已成功处理并保存到对应域的文件夹中。")
    else:
        print(f"\n错误 处理过程中遇到问题，请检查错误信息")

if __name__ == "__main__":
    main()