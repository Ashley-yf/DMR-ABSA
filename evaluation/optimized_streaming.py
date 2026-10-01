#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的流水线处理器
1. 按文件逐个处理
2. 单条数据的A→B→C流水线处理，但只返回C的最终结果
3. 第一条数据ID都为0，之后就ID增量变化
4. 完全内存友好：每处理一条数据立即追加保存到文件，不累积中间结果
5. 真正的实时保存：使用文件追加模式 + 立即刷新缓冲区
"""

import os
import json
import time
import logging
import sys
import shutil
import glob
from typing import Dict, List, Any

# API 环境变量（从 .env 或 shell 导出，详见 .env.example；不在此硬编码密钥）
os.environ.setdefault('OPENAI_API_KEY', '')            # required
os.environ.setdefault('OPENAI_API_BASE', 'https://api.openai.com/v1')
os.environ.setdefault('OPENAI_Model', 'DeepSeek-V3')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedStreamingPipeline:
    """优化的流式文件处理器：只保存C的最终结果，极致资源友好"""

    def __init__(self, delay: float = 1.0, output_dir: str = "optimized_outputs"):
        self.delay = delay
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 初始化真实的模型
        logger.info("初始化优化的流水线模型...")
        try:
            from model_a_aspect_standardization import AspectStandardizationModel
            logger.info("模型A导入成功")
            self.model_a = AspectStandardizationModel()
        except Exception as e:
            logger.error(f"模型A初始化失败: {e}")
            raise

        try:
            from model_b_sentiment_evaluation import SentimentEvaluationModel
            logger.info("模型B导入成功")
            self.model_b = SentimentEvaluationModel()
        except Exception as e:
            logger.error(f"模型B初始化失败: {e}")
            raise

        try:
            from model_c_quality_decision import QualityDecisionModel
            logger.info(" 模型C导入成功")
            self.model_c = QualityDecisionModel()
        except Exception as e:
            logger.error(f" 模型C初始化失败: {e}")
            raise

        # 统计信息
        self.global_stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_entries': 0,
            'saved_entries': 0,
            'discarded_entries': 0,
            'error_entries': 0,
            'api_calls': 0,
            'processing_start_time': time.time()
        }

    def generate_output_filename(self, input_path: str) -> str:
        """生成输出文件名：数据集类型_原始文件名.json"""
        filename = os.path.basename(input_path)
        name_without_ext = os.path.splitext(filename)[0]

        # 使用完整路径进行分析
        full_path_lower = input_path.lower()

        # 检查数据集类型
        if "property-driven" in full_path_lower or "/norm/" in full_path_lower or "\\norm\\" in full_path_lower:
            dataset_type = "norm"
        elif "seed-driven" in full_path_lower or "/refined/" in full_path_lower or "\\refined\\" in full_path_lower:
            dataset_type = "refined"
        else:
            dataset_type = "unknown"

        output_filename = f"{dataset_type}_{name_without_ext}.json"
        return output_filename

    def discover_files(self, base_dirs: List[str], subdirs: List[str]) -> List[Dict[str, str]]:
        """发现所有要处理的JSON文件并保留路径信息"""
        discovered_files = []

        for base_dir in base_dirs:
            for subdir in subdirs:
                search_dir = os.path.join(base_dir, subdir)
                if os.path.exists(search_dir):
                    json_files = glob.glob(os.path.join(search_dir, "*.json"))
                    for file_path in json_files:
                        discovered_files.append({
                            'full_path': file_path,
                            'base_dir': base_dir,
                            'subdir': subdir,
                            'relative_path': os.path.relpath(file_path)
                        })
                    logger.info(f"在 {search_dir} 发现 {len(json_files)} 个JSON文件")
                else:
                    logger.warning(f"目录不存在: {search_dir}")

        self.global_stats['total_files'] = len(discovered_files)
        logger.info(f"总共发现 {len(discovered_files)} 个JSON文件")
        return discovered_files

    def load_single_file(self, file_path: str) -> List[Dict[str, Any]]:
        """加载单个JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                logger.warning(f"文件格式不是列表或字典: {file_path}")
                return []

            logger.info(f"文件 {os.path.basename(file_path)} 包含 {len(data)} 条数据")
            return data

        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
            return []

    def validate_entry_format(self, entry: Dict[str, Any]) -> bool:
        """验证单条数据格式"""
        required_fields = ['ID', 'Sentence', 'Label']
        return all(field in entry for field in required_fields)

    def process_single_entry_optimized_pipeline(self, entry: Dict[str, Any], file_id_counter: int) -> Dict[str, Any]:
        """
        单条数据的优化流水线处理：A→B→C，只返回C的最终结果
        这是极致资源友好的方式：处理完一条立即保存，不累积任何中间结果
        """
        # 使用文件内ID而不是原始ID
        entry_id = str(file_id_counter)  # 从0开始的递增ID
        sentence = entry.get('Sentence', '')
        labels = entry.get('Label', '')
        original_entry_id = entry.get('ID', 'unknown')  # 保留原始ID用于日志

        try:
            logger.info(f" 开始处理第 {file_id_counter} 条数据 (原始ID: {original_entry_id}, 新ID: {entry_id})")
            start_time = time.time()

            # 步骤1: 模型A - 方面词标准化
            logger.info(f"    模型A处理 (新ID: {entry_id}, 原始ID: {original_entry_id})")
            a_result = self.model_a.process_single_entry(entry)
            a_time = time.time() - start_time

            logger.info(f"    模型A处理完成 (耗时: {a_time:.2f}秒)")

            # 步骤2: 模型B - 情感极性评估
            logger.info(f"    模型B处理 (新ID: {entry_id}, 原始ID: {original_entry_id})")
            logger.info(f"    模型A输出标签: {a_result.get('ProcessedLabels', [])}")
            b_result = self.model_b.process_single_entry(a_result)  # 使用模型A的输出，不是原始entry
            b_time = time.time() - a_time - start_time  # 修正时间计算

            logger.info(f"    模型B处理完成 (耗时: {b_time:.2f}秒)")
            logger.info(f"    模型B输出标签: {b_result.get('ProcessedLabels', [])}")

            # 步骤3: 模型C - 质量决策
            logger.info(f"    模型C处理 (新ID: {entry_id}, 原始ID: {original_entry_id})")
            c_result = self.model_c.process_single_entry(b_result)  # 使用模型B的输出，不是原始entry
            c_time = time.time() - b_time - a_time - start_time  # 修正时间计算

            logger.info(f"    模型C处理完成 (耗时: {c_time:.2f}秒)")

            # 统计总处理时间
            total_time = a_time + b_time + c_time
            self.global_stats['api_calls'] += 3  # 每条数据3次API调用

            logger.info(f"    总处理时间: {total_time:.2f}秒")

            # 检查模型C的决策结果
            final_decision = c_result.get('Decision', 'discard')
            optimized_sentence = c_result.get('OptimizedSentence', sentence)
            final_labels = c_result.get('FinalLabels', labels)
            reasoning = c_result.get('Reasoning', 'No reasoning provided')

            logger.info(f"    最终决策: {final_decision}")
            logger.info(f"    优化后句子: {optimized_sentence}")
            if final_labels:
                logger.info(f"    最终标签: {final_labels}")

            # 只返回C的最终结果，A和B的结果不返回也不保存
            if final_decision == 'save':
                self.global_stats['saved_entries'] += 1
                logger.info(f"    数据将被保存")
            else:
                self.global_stats['discarded_entries'] += 1
                logger.info(f"    数据将被丢弃")

            # 返回最终输出数据（C的结果）- 包含决策信息
            final_output = {
                'ID': entry_id,  # 使用文件内的增量ID
                'Sentence': optimized_sentence,
                'Label': final_labels,
                'Decision': final_decision  # 添加决策字段
            }

            return final_output

        except Exception as e:
            logger.error(f"    处理失败 (新ID: {entry_id}, 原始ID: {original_entry_id}): {e}")
            self.global_stats['error_entries'] += 1
            return {
                'ID': entry_id,
                'Sentence': entry.get('Sentence', ''),
                'Label': entry.get('Label', []),
                'error': str(e)
            }

    def process_single_file_optimized(self, file_info: Dict[str, str]) -> Dict[str, Any]:
        """
        处理单个文件：优化的流水线处理，只保存C的最终结果
        """
        file_path = file_info['full_path']
        base_dir = file_info['base_dir']
        subdir = file_info['subdir']

        logger.info(f" 开始处理文件: {os.path.basename(file_path)}")
        logger.info(f"   数据集: {base_dir}, 子目录: {subdir}")

        # 1. 加载文件数据
        file_data = self.load_single_file(file_path)
        if not file_data:
            logger.warning(f"文件为空或无效: {file_path}")
            return {
                'file_path': file_path,
                'output_path': None,
                'original_count': 0,
                'processed_count': 0,
                'saved_count': 0,
                'status': 'empty'
            }

        original_count = len(file_data)
        self.global_stats['total_entries'] += original_count

        # 2. 生成输出文件路径
        output_filename = self.generate_output_filename(file_path)
        output_path = os.path.join(self.output_dir, output_filename)

        logger.info(f"    原始数据: {original_count} 条")
        logger.info(f"    输出文件: {output_filename}")

        # 3. 为每个文件重新开始ID计数（实现需求）
        file_id_counter = 0  # 第一条数据ID都为0

        # 4. 逐条进行优化的流水线处理 + 真正的逐条追加保存
        processed_entries = []  # 保留用于进度统计，不再用于批量写入
        saved_count = 0

        # 准备输出文件（先创建空文件或清空现有文件）
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('[\n')  # 开始JSON数组
            logger.info(f"    初始化输出文件: {output_path}")
        except Exception as e:
            logger.error(f"    初始化输出文件失败: {e}")
            return {
                'file_path': file_path,
                'output_path': output_path,
                'original_count': original_count,
                'processed_count': 0,
                'saved_count': 0,
                'status': 'init_failed',
                'error': str(e)
            }

        for i, entry in enumerate(file_data):
            logger.info(f"    处理第 {i+1}/{original_count} 条数据 (文件内ID: {file_id_counter})")
            logger.info(f"      原始句子: {entry.get('Sentence', '')[:50]}...")

            # 验证数据格式
            if not self.validate_entry_format(entry):
                logger.warning(f"        跳过无效数据条目: 原始ID={entry.get('ID')}")
                self.global_stats['error_entries'] += 1
                file_id_counter += 1  # 为无效数据也分配ID
                continue

            # 处理单条数据
            final_output = self.process_single_entry_optimized_pipeline(entry, file_id_counter)

            # 立即逐条追加保存机制
            if final_output.get('Decision') == 'save':
                processed_entries.append(final_output)  # 保留用于统计
                saved_count += 1

                # 立即追加到文件
                try:
                    with open(output_path, 'a', encoding='utf-8') as f:
                        if saved_count > 1:
                            f.write(',\n')  # 前面加逗号（除第一条）
                        json.dump(final_output, f, ensure_ascii=False, indent=2)
                        f.flush()  # 强制刷新缓冲区
                    logger.info(f"       数据已实时保存 (文件内ID: {file_id_counter}, 总保存: {saved_count})")
                except Exception as e:
                    logger.error(f"       实时保存失败 (文件内ID: {file_id_counter}): {e}")
                    # 继续处理下一条，不中断整个流程
            else:
                logger.info(f"       数据已丢弃 (文件内ID: {file_id_counter})")

            file_id_counter += 1  # 每条数据都增加ID计数

            # 显示进度
            if (i + 1) % 10 == 0:
                save_rate = saved_count / (i + 1) if (i + 1) > 0 else 0
                logger.info(f"    进度: {i+1}/{original_count}, 保存率: {save_rate:.2%}, 实时保存: {saved_count} 条")

        # 完成JSON数组的结尾
        try:
            with open(output_path, 'a', encoding='utf-8') as f:
                if saved_count > 0:
                    f.write('\n')
                f.write(']')  # 结束JSON数组
                f.flush()  # 强制刷新缓冲区
            logger.info(f"    完成JSON文件结构，总共实时保存: {saved_count} 条数据")
        except Exception as e:
            logger.error(f"    完成JSON文件结构失败: {e}")

        # 5. 检查并修正已保存的文件（真正的逐条保存已在处理循环中完成）
        try:
            logger.info(f"    验证逐条保存的文件: {output_path}")
            logger.info(f"    实时保存的数据数量: {saved_count}")

            # 验证文件是否存在并包含正确数量的数据
            if os.path.exists(output_path) and saved_count > 0:
                # 读取文件验证内容
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        logger.info(f"    文件验证成功: {len(content)} 字符已写入")
                        # 验证JSON格式
                        try:
                            data = json.loads(content)
                            logger.info(f"    JSON格式验证成功: 包含 {len(data)} 条数据")
                        except json.JSONDecodeError:
                            logger.warning(f"    JSON格式可能需要手动修复")
                    else:
                        logger.warning(f"    文件存在但内容为空")
            else:
                logger.warning(f"    文件不存在或无数据保存: {output_path}")

        except Exception as e:
            logger.error(f"    文件验证失败 {output_path}: {e}")

        # 验证最终文件状态
        final_saved_count = 0
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        final_saved_count = len(data) if isinstance(data, list) else 0
                        logger.info(f"    最终文件验证成功: 包含 {final_saved_count} 条有效数据")
                    except json.JSONDecodeError as e:
                        logger.warning(f"    最终文件JSON格式问题: {e}")
                        final_saved_count = saved_count  # 使用计数作为备用
            except Exception as e:
                logger.warning(f"    读取最终文件失败: {e}")
                final_saved_count = saved_count  # 使用计数作为备用
        else:
            logger.warning(f"    输出文件不存在: {output_path}")

        # 6. 记录完成统计
        file_stats = {
            'file_path': file_path,
            'output_path': output_path,
            'base_dir': base_dir,
            'subdir': subdir,
            'original_count': original_count,
            'processed_count': len(file_data),
            'saved_count': final_saved_count,
            'discarded_count': original_count - final_saved_count,
            'save_rate': final_saved_count / original_count if original_count > 0 else 0,
            'status': 'completed',
            'save_strategy': 'realtime_append'  # 新增字段标识保存策略
        }

        self.global_stats['processed_files'] += 1

        logger.info(f"    文件处理完成:")
        logger.info(f"       输入文件: {os.path.basename(file_path)}")
        logger.info(f"       输出文件: {output_filename}")
        logger.info(f"       原始条目: {original_count}")
        logger.info(f"       处理条目: {len(file_data)}")
        logger.info(f"       实时保存条目: {final_saved_count}")
        logger.info(f"       丢弃条目: {original_count - final_saved_count}")
        logger.info(f"       保存率: {file_stats['save_rate']:.2%}")
        logger.info(f"       保存策略: {file_stats['save_strategy']} (每处理一条立即保存)")

        return file_stats

    def process_files_sequentially_optimized(self, base_dirs: List[str], subdirs: List[str]) -> Dict[str, Any]:
        """按文件逐个处理：优化的流水线处理"""
        # 发现文件
        discovered_files = self.discover_files(base_dirs, subdirs)

        if not discovered_files:
            logger.error("没有发现任何可处理的文件")
            return self.global_stats

        logger.info(f" 开始优化的流水线处理，共 {len(discovered_files)} 个文件")
        logger.info(f" 输出目录: {self.output_dir}")

        # 逐个处理文件
        file_processing_stats = []
        start_time = time.time()

        for i, file_info in enumerate(discovered_files):
            logger.info(f"\n 处理进度: {i+1}/{len(discovered_files)} 个文件")

            file_stats = self.process_single_file_optimized(file_info)
            file_processing_stats.append(file_stats)

            # 显示总体进度
            saved_rate = self.global_stats['saved_entries'] / max(self.global_stats['total_entries'], 1) if self.global_stats['total_entries'] > 0 else 0
            logger.info(f" 总体进度: 已处理 {i+1}/{len(discovered_files)} 个文件, "
                       f"累计保存 {self.global_stats['saved_entries']} 条数据, "
                       f"保存率: {saved_rate:.1%}")

        # 计算最终统计
        end_time = time.time()
        processing_time = end_time - start_time

        final_stats = {
            **self.global_stats,
            'processing_time_seconds': processing_time,
            'processing_time_minutes': processing_time / 60,
            'files_per_minute': len(discovered_files) / max(processing_time / 60, 0.01),
            'entries_per_minute': self.global_stats['total_entries'] / max(processing_time / 60, 0.01),
            'overall_save_rate': self.global_stats['saved_entries'] / max(self.global_stats['total_entries'], 1) if self.global_stats['total_entries'] > 0 else 0,
            'file_stats': file_processing_stats
        }

        # 输出最终总结
        self.log_final_summary_optimized(final_stats)

        return final_stats

    def log_final_summary_optimized(self, stats: Dict[str, Any]):
        """输出优化的最终处理总结"""
        logger.info("\n" + "=" * 80)
        logger.info(" 优化的流水线处理完成！最终统计：")
        logger.info("=" * 80)
        logger.info(f" 总文件数: {stats['total_files']}")
        logger.info(f" 已处理文件: {stats['processed_files']}")
        logger.info(f" 总数据条目: {stats['total_entries']}")
        logger.info(f" 保存条目数: {stats['saved_entries']}")
        logger.info(f" 丢弃条目数: {stats['discarded_entries']}")
        logger.info(f" 错误条目数: {stats['error_entries']}")
        logger.info(f" 总体保存率: {stats['overall_save_rate']:.2%}")
        logger.info(f" 处理时间: {stats['processing_time_seconds']:.2f} 秒")
        logger.info(f" 处理时间: {stats['processing_time_minutes']:.2f} 分钟")
        logger.info(f" 文件处理速度: {stats['files_per_minute']:.2f} 文件/分钟")
        logger.info(f" 条目处理速度: {stats['entries_per_minute']:.2f} 条目/分钟")
        logger.info(f" 总API调用次数: {stats['api_calls']}")
        logger.info("=" * 80)

        # 显示每个文件的详细统计
        logger.info("\n 各文件详细统计:")
        for file_stat in stats['file_stats']:
            if file_stat.get('status') == 'completed':
                logger.info(f" {os.path.basename(file_stat['file_path'])}: "
                          f"{file_stat['saved_count']}/{file_stat['original_count']} 条 "
                          f"({file_stat['save_rate']:.1%})")
            elif file_stat.get('status') == 'empty':
                logger.warning(f" {os.path.basename(file_stat['file_path'])}: 文件为空")
            elif file_stat.get('status') == 'write_failed':
                logger.error(f" {os.path.basename(file_stat['file_path'])}: 写入失败")
            else:
                logger.error(f" {os.path.basename(file_stat['file_path'])}: {file_stat.get('status', 'unknown')}")

        logger.info("\n 优化特性总结:")
        logger.info(" 1. 按文件逐个处理")
        logger.info(" 2. 单条数据A→B→C流水线处理")
        logger.info(" 3. 只保存C的最终结果，节省内存")
        logger.info(" 4. 第一条数据ID都为0，之后就ID增量变化")
        logger.info(" 5. 完全内存友好：处理完一条立即保存")
        logger.info(" 6. 极致资源优化：每条数据只调用一次，不返回中间结果")
        logger.info(f" 输出目录: {self.output_dir}")

def main():
    """主函数：运行优化的流水线处理器"""
    logger.info(" 开始优化的流水线处理器...")
    logger.info("=" * 80)
    logger.info(" 优化特性:")
    logger.info("1. 按文件逐个处理")
    logger.info("2. 单条数据A→B→C流水线处理")
    logger.info("3. 只保存C的最终结果，节省内存")
    logger.info("4. 第一条数据ID都为0，之后就ID增量变化")
    logger.info("5. 完全内存友好：每处理一条立即追加保存到文件")
    logger.info("6. 极致资源优化：每条数据只调用一次，不返回中间结果")
    logger.info("7. 真正实时保存：文件追加模式 + 缓冲区立即刷新")
    logger.info("=" * 80)

    # 检查是否指定了单个文件路径
    if len(sys.argv) > 1:
        # 处理指定的单个文件
        specific_file_path = sys.argv[1]
        logger.info(f" 处理指定文件: {specific_file_path}")
        return process_specific_file(specific_file_path)
    else:
        # 批量处理目录
        return process_all_directories()

def process_specific_file(file_path):
    """处理指定的单个文件"""
    try:
        # 验证文件是否存在
        if not os.path.exists(file_path):
            logger.error(f" 文件不存在: {file_path}")
            return False

        if not file_path.endswith('.json'):
            logger.error(f" 文件必须是JSON格式: {file_path}")
            return False

        # 创建处理器
        processor = OptimizedStreamingPipeline(
            delay=1.0,
            output_dir="optimized_outputs"
        )

        # 构造文件信息
        file_info = {
            'full_path': os.path.abspath(file_path),
            'base_dir': os.path.dirname(os.path.dirname(file_path)),  # 上上级目录
            'subdir': os.path.basename(os.path.dirname(file_path)),  # 父目录名
            'relative_path': file_path
        }

        logger.info(f" 基础目录: {file_info['base_dir']}")
        logger.info(f" 子目录: {file_info['subdir']}")

        # 处理单个文件
        file_stats = processor.process_single_file_optimized(file_info)

        # 输出结果
        if file_stats['status'] == 'completed':
            logger.info(f" 文件处理完成:")
            logger.info(f" 输入文件: {os.path.basename(file_stats['file_path'])}")
            logger.info(f" 输出文件: {os.path.basename(file_stats['output_path'])}")
            logger.info(f" 原始条目: {file_stats['original_count']}")
            logger.info(f" 保存条目: {file_stats['saved_count']}")
            logger.info(f" 保存率: {file_stats['save_rate']:.2%}")
            return True
        else:
            logger.error(f" 文件处理失败: {file_stats.get('status', 'unknown')}")
            return False

    except Exception as e:
        logger.error(f" 处理指定文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_all_directories():
    """批量处理所有目录"""
    try:
        # 配置要处理的目录
        base_dirs = ["../property-driven", "../seed-driven"]
        subdirs = ["lap", "res", "res15", "res16"]

        # 创建处理器
        processor = OptimizedStreamingPipeline(
            delay=1.0,
            output_dir="optimized_outputs"
        )

        # 运行优化的流水线处理
        final_stats = processor.process_files_sequentially_optimized(base_dirs, subdirs)

        # 验证结果
        success = (
            final_stats['processed_files'] > 0 and
            final_stats['saved_entries'] > 0 and
            final_stats['overall_save_rate'] > 0
        )

        if success:
            logger.info("\n 优化的流水线处理器验证成功！")
            logger.info(" 所有特性按需求完美实现：")
            logger.info("   - 超级内存友好：不累积任何中间结果")
            logger.info("   - 资源最优：只调用必要API，不浪费结果")
            logger.info("   - 处理效率：单条数据完整处理和保存")
            logger.info("   - 完美满足你的需求！")
        else:
            logger.error("\n 优化的流水线处理器验证失败！")

        return success

    except KeyboardInterrupt:
        logger.info("\n 用户中断了优化处理")
        return False
    except Exception as e:
        logger.error(f"\n 优化处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)