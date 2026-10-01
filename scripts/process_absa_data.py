#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import argparse
import ast


class ABSADataProcessor:
    """
    Processor for ABSA dataset generation from prompt files without LLM calls

    成功 完全去除大模型调用，节省资源
    成功 直接通过代码解析和提取数据
    成功 支持两种提示词格式的解析
    """

    def __init__(self, model_short_name: str = "direct"):
        """
        Initialize the processor

        Args:
            model_short_name: Short name for output files
        """
        self.model_short_name = model_short_name
        self.processing_stats = {
            "total_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "missing_aspects": 0
        }

    def parse_keypoint_prompt_gen(self, prompt_gen: str) -> Tuple[str, List[List[str]]]:
        """
        Parse keypoint prompt_gen string to extract sentence and labels

        搜索 Keypoint Prompts 解析逻辑：
        - 从 prompt_gen 字段提取数据
        - 按 "llm_output:" 分割获取生成的句子
        - 从基础提示词中提取 Label: 信息
        - 使用正则表达式和安全解析提取标签

        Args:
            prompt_gen: String containing sentence and labels from keypoint format

        Returns:
            Tuple of (sentence, labels) where labels is list of [aspect, polarity]
        """
        try:
            # Split by "llm_output:" to get the LLM generated sentence
            if "llm_output:" in prompt_gen:
                parts = prompt_gen.split("llm_output:")
                base_prompt = parts[0]
                llm_output = parts[1].strip()

                # Extract sentence from llm_output
                sentence = llm_output.strip()

                # Extract labels from the base prompt - look for Label: pattern
                if "Label:" in base_prompt:
                    # Get the last occurrence of "Label:" in the base prompt
                    label_matches = re.findall(r'Label:\s*(\[[^\]]+\])', base_prompt)
                    if label_matches:
                        label_str = label_matches[-1]  # Use the last match
                        try:
                            labels = ast.literal_eval(label_str)
                            if isinstance(labels, list) and all(isinstance(l, list) and len(l) == 2 for l in labels):
                                return sentence, labels
                        except (ValueError, SyntaxError):
                            pass

                # Fallback: try to find any pattern that looks like labels
                label_pattern = r'\[\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*\]'
                matches = re.findall(label_pattern, base_prompt)
                if matches:
                    labels = [[match[0], match[1]] for match in matches]
                    return sentence, labels

            # If no llm_output found, try original parsing as fallback
            prompt_gen = prompt_gen.replace('\\n', '\n')

            if "Label:" in prompt_gen:
                sentence_part = prompt_gen.split("Label:")[0].strip()
                if sentence_part.startswith("Sentence:"):
                    sentence = sentence_part.replace("Sentence:", "").strip()
                else:
                    sentence = sentence_part

                label_part = prompt_gen.split("Label:")[1].strip()
                try:
                    labels = ast.literal_eval(label_part)
                    if isinstance(labels, list) and all(isinstance(l, list) and len(l) == 2 for l in labels):
                        return sentence, labels
                except (ValueError, SyntaxError):
                    pass

            return sentence, []

        except Exception as e:
            print(f"Error parsing keypoint prompt_gen: {e}")
            return prompt_gen, []

    def parse_instance_prompt_aug(self, prompt_aug: str) -> Tuple[str, List[List[str]]]:
        """
        Parse instance prompt_aug string to extract sentence and labels

        搜索 Instance Prompts 解析逻辑：
        - 从 prompt_aug 字段提取数据
        - 查找 "4 Diverse ... Sentences with Labels:\n\n1. Sentence:" 模式
        - 提取第一个句子作为生成的内容
        - 从原始提示词部分提取标签信息

        Args:
            prompt_aug: String containing sentence and labels from instance format

        Returns:
            Tuple of (sentence, labels) where labels is list of [aspect, polarity]
        """
        try:
            # Look for the specific pattern for instance prompts
            pattern = r'4 Diverse .*Sentences with Labels:\n\n1\. Sentence:\s*(.+?)(?=\n|$)'

            match = re.search(pattern, prompt_aug, re.DOTALL | re.IGNORECASE)
            if match:
                sentence = match.group(1).strip()

                # Extract labels from the original prompt part (before the pattern)
                original_part = prompt_aug.split(match.group(0))[0]

                # Look for label patterns in the original part
                label_pattern = r'Label:\s*(\[[^\]]+\])'
                label_matches = re.findall(label_pattern, original_part)

                if label_matches:
                    try:
                        labels = ast.literal_eval(label_matches[-1])  # Use the last match
                        if isinstance(labels, list) and all(isinstance(l, list) and len(l) == 2 for l in labels):
                            return sentence, labels
                    except (ValueError, SyntaxError):
                        pass

                # Alternative pattern: find all [aspect, polarity] patterns
                alt_pattern = r'\[\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*\]'
                matches = re.findall(alt_pattern, original_part)
                if matches:
                    labels = [[match[0], match[1]] for match in matches]
                    return sentence, labels

            # Fallback: try to extract any sentence and find labels
            lines = prompt_aug.split('\n')
            sentence = ""
            labels = []

            for line in lines:
                line = line.strip()
                if line.startswith("Sentence:"):
                    sentence = line.replace("Sentence:", "").strip()
                elif line.startswith("1. Sentence:"):
                    sentence = line.replace("1. Sentence:", "").strip()
                elif "Label:" in line:
                    label_part = line.split("Label:")[1].strip()
                    try:
                        extracted_labels = ast.literal_eval(label_part)
                        if isinstance(extracted_labels, list):
                            labels = extracted_labels
                    except (ValueError, SyntaxError):
                        pass

            return sentence, labels

        except Exception as e:
            print(f"Error parsing instance prompt_aug: {e}")
            return prompt_aug, []


    def process_single_item(self, item: Dict[str, Any], prompt_type: str) -> Optional[Dict[str, Any]]:
        """
        Process a single item from the dataset

        Args:
            item: Dictionary containing ID and prompt data
            prompt_type: "keypoint" or "instance"

        Returns:
            Processed result or None if failed
        """
        try:
            # Extract basic information
            item_id = str(item.get("ID", ""))

            if prompt_type == "keypoint":
                prompt_data = item.get("prompt_gen", "")
                sentence, labels = self.parse_keypoint_prompt_gen(prompt_data)
            elif prompt_type == "instance":
                prompt_data = item.get("prompt_aug", "")
                sentence, labels = self.parse_instance_prompt_aug(prompt_data)
            else:
                print(f"Unknown prompt type: {prompt_type}")
                return None

            if not sentence:
                print(f"No sentence found for ID {item_id}")
                return None

            if not labels:
                print(f"No labels found for ID {item_id}")
                return None

            # Create result in standard format (remove aspects field as requested)
            result = {
                "ID": item_id,
                "Sentence": sentence,  # Capitalize 'Sentence'
                "Label": labels        # Capitalize 'Label'
            }

            self.processing_stats["successful_extractions"] += 1
            return result

        except Exception as e:
            print(f"Error processing item {item.get('ID', 'unknown')}: {e}")
            self.processing_stats["failed_extractions"] += 1
            return None

    def extract_domain_from_filename(self, filename: str) -> str:
        """
        Extract domain from filename (text before the first underscore)

        Args:
            filename: Filename like "res_sample2.json"

        Returns:
            Domain name like "res"
        """
        base_name = os.path.splitext(filename)[0]  # Remove extension
        if "_" in base_name:
            return base_name.split("_")[0]
        else:
            return base_name

    def process_file(self, input_file: str, output_base_dir: str, prompt_type: str = "keypoint", source_filename: str = None):
        """
        Process a single JSON file

        Args:
            input_file: Path to JSON file
            output_base_dir: Base output directory (property-driven)
            prompt_type: "keypoint" or "instance"
            source_filename: Source filename (for report naming)
        """
        # Extract domain and filename from input file
        filename = os.path.basename(input_file)
        base_filename = os.path.splitext(filename)[0]
        domain = self.extract_domain_from_filename(base_filename)

        # Use provided source_filename or extract from input file
        final_source_filename = source_filename or base_filename

        print(f"Processing file: {filename}")
        print(f"Domain: {domain}")
        print(f"Prompt type: {prompt_type}")

        # Create domain directory if it doesn't exist
        domain_dir = os.path.join(output_base_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)

        # Output filename (matching input filename)
        output_filename = f"{self.model_short_name}_{base_filename}.json"
        output_file = os.path.join(domain_dir, output_filename)

        # Load input data
        print(f"Loading data from {input_file}...")
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        print(f"Loaded {len(data)} items")

        # Load existing results if output file exists
        existing_results = []
        processed_ids = set()

        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8-sig') as f:
                    existing_results = json.load(f)
                    processed_ids = set(item.get("ID", "") for item in existing_results)
                print(f"Loaded {len(existing_results)} existing results")
            except Exception as e:
                print(f"Warning: Could not load existing results from {output_file}: {e}")

        # Process each item
        new_results = []
        processed_count = 0
        failed_count = 0
        skipped_count = 0

        for i, item in enumerate(data):
            item_id = str(item.get("ID", ""))

            # Skip if already processed
            if item_id in processed_ids:
                skipped_count += 1
                continue

            print(f"\nProgress: {i+1}/{len(data)} (New: {processed_count+1}, Failed: {failed_count}, Skipped: {skipped_count})")
            print(f"Processing ID {item_id}...")

            result = self.process_single_item(item, prompt_type)

            if result:
                new_results.append(result)
                processed_count += 1
                print(f"   [SUCCESS] Successfully processed ID {item_id}")
                print(f"   Sentence: {result['sentence'][:80]}...")
                print(f"   Labels: {result['label']}")

                # Save intermediate results every 10 items
                if len(new_results) % 10 == 0:
                    self.save_results(existing_results + new_results, output_file)
                    print(f"   [SAVED] Saved intermediate results ({len(existing_results + new_results)} total)")
            else:
                failed_count += 1
                print(f"   [FAILED] Failed to process ID {item_id}")

        # Save final results
        if new_results:
            final_results = existing_results + new_results
            self.save_results(final_results, output_file)
            print(f"\nSaved final results to {output_file}")

        # Print statistics
        print(f"\nProcessing complete for {filename}!")
        print(f"Domain: {domain}")
        print(f"Prompt type: {prompt_type}")
        print(f"Total items: {len(data)}")
        print(f"Already processed: {skipped_count}")
        print(f"Successfully processed: {processed_count}")
        print(f"Failed: {failed_count}")
        print(f"Total results in file: {len(existing_results + new_results)}")
        print(f"Missing aspects: {self.processing_stats['missing_aspects']}")

        # Reset stats for next file
        self.processing_stats = {
            "total_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "missing_aspects": 0
        }

    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """
        Save results to JSON file

        Args:
            results: List of processed results
            output_file: Output file path
        """
        # Sort results by ID for consistency
        results.sort(key=lambda x: int(x.get("ID", "0")) if x.get("ID", "0").isdigit() else x.get("ID", ""))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    """
    Main function to run the processor

    启动 运行方式：
    # 运行完整的批量处理
    python scripts/process_all_datasets.py

    # 或单独处理文件（在仓库根目录执行）
    python scripts/process_absa_data.py input_file output_dir source_filename --prompt_type keypoint
    python scripts/process_absa_data.py input_file output_dir source_filename --prompt_type instance
    """
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process ABSA prompt files without LLM calls')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('output_base_dir', help='Output base directory')
    parser.add_argument('source_filename', help='Source filename for reports')
    parser.add_argument('--prompt_type', choices=['keypoint', 'instance'],
                       help='Type of prompts (keypoint or instance)', required=True)

    args = parser.parse_args()

    print(f"Processing file:")
    print(f"  Input file: {args.input_file}")
    print(f"  Output directory: {args.output_base_dir}")
    print(f"  Source filename: {args.source_filename}")
    print(f"  Prompt type: {args.prompt_type}")

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return

    # Create processor
    try:
        processor = ABSADataProcessor()
    except Exception as e:
        print(f"Error creating processor: {e}")
        return

    # Process single file
    processor.process_file(
        input_file=args.input_file,
        output_base_dir=args.output_base_dir,
        prompt_type=args.prompt_type,
        source_filename=args.source_filename
    )


if __name__ == "__main__":
    main()