"""
模型C：句子质量提升与最终决策模型
角色：句子优化与质量评估专家，负责提升句子表达并最终决定是否保存数据
"""

import json
import time
import logging
import sys
import os
from typing import Dict, List, Tuple, Any

# 添加父目录到路径以导入qnaigc_api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_prompt.qnaigc_api import create_qnaigc_client

class QualityDecisionModel:
    def __init__(self):
        """
        初始化句子质量提升与最终决策模型
        """
        self.client = create_qnaigc_client()

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # 实时保存相关状态
        self._incremental_file_state = {}  # 存储每个输出文件的状态

        self.system_prompt = """You are a sentence optimization and quality assessment expert, responsible for improving sentence expression and deciding whether to save sentiment analysis data. Given a data entry containing id, sentence, and label (aspect terms and sentiment polarity), along with processing status from the previous two models, your task is to:

1. First, evaluate the sentence quality. If the sentence has the following issues, optimize it:
   - Grammar errors or awkward phrasing
   - Unclear expression that may cause ambiguity
   - Overly verbose or wordy, can be simplified
   The optimized sentence must maintain the original meaning and cannot change the relationship between aspect terms and sentiment polarity.

2. If the sentence is already clear and error-free, keep it as is.

3. Then, based on the optimized sentence and label, evaluate the overall data quality. High-quality data should meet:
   - Aspect terms are accurate and consistent with sentence context (judged by standardization_status)
   - Sentiment polarity is correct (judged by sentiment_status)
   - Sentence is clear, unambiguous, and grammatically correct (after optimization)

4. Check if data is "consistently present": if data content is empty, repetitive (with common patterns), or obviously low quality (such as irrelevant aspect terms, contradictory sentiment polarity), then do not save.

5. Only save high-quality data. If data was modified by Model A or Model B (standardization_status or sentiment_status is "modified"), but final quality is high, it can still be saved. Similarly, if the sentence was optimized but final quality is high, it can also be saved.

6. Output the save decision and the final data entry (including optimized sentence and label).

Quality standards:
- Sentence optimization must maintain original meaning, especially not changing aspect term references and sentiment tendency
- Save condition: aspect terms and sentiment polarity are both accurate, sentence is clear (possibly optimized), and non-repetitive low-quality content
- Discard condition: data is consistently present (such as duplicate entries), aspect terms are irrelevant, sentiment polarity is wrong, sentence cannot be optimized to be readable

Please strictly output in the following JSON format:
{
    "optimized_sentence": "optimized sentence (if needed)",
    "sentence_optimized": true/false,
    "decision": "save/discard",
    "quality_score": 1-10,
    "reasoning": "detailed reasoning"
}"""

    def process_single_entry(self, data_entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个数据条目（模型B的输出）

        Args:
            data_entry: 包含ID, Sentence, ProcessedLabels的数据字典

        Returns:
            处理后的数据字典，包含最终决策
        """
        try:
            # 提取数据
            entry_id = data_entry.get("ID", "")
            sentence = data_entry.get("Sentence", "")
            processed_labels = data_entry.get("ProcessedLabels", [])
            original_labels = data_entry.get("OriginalLabels", [])

            # 检查是否有修改
            has_modifications = any(
                label.get("standardization_status") == "modified" or
                label.get("sentiment_status") == "modified"
                for label in processed_labels
            )

            # 调用模型进行质量评估和决策
            decision_result = self._evaluate_quality_and_decide(
                sentence, processed_labels, has_modifications
            )

            # 准备最终的标签格式
            final_labels = [
                [label.get("aspect", ""), label.get("polarity", "")]
                for label in processed_labels
            ]

            return {
                "ID": entry_id,
                "OriginalSentence": sentence,
                "OptimizedSentence": decision_result["optimized_sentence"],
                "SentenceOptimized": decision_result["sentence_optimized"],
                "FinalLabels": final_labels,
                "Decision": decision_result["decision"],
                "QualityScore": decision_result["quality_score"],
                "Reasoning": decision_result["reasoning"],
                "ProcessingHistory": {
                    "original_labels": original_labels,
                    "processed_labels": processed_labels,
                    "has_modifications": has_modifications
                }
            }

        except Exception as e:
            self.logger.error(f"处理数据条目时出错: {e}")
            return {
                "ID": data_entry.get("ID", ""),
                "OriginalSentence": data_entry.get("Sentence", ""),
                "OptimizedSentence": data_entry.get("Sentence", ""),
                "SentenceOptimized": False,
                "FinalLabels": [],
                "Decision": "discard",
                "QualityScore": 0,
                "Reasoning": f"处理错误: {str(e)}",
                "ProcessingHistory": {
                    "original_labels": data_entry.get("OriginalLabels", []),
                    "processed_labels": data_entry.get("ProcessedLabels", []),
                    "has_modifications": False,
                    "error": str(e)
                }
            }

    def _evaluate_quality_and_decide(self, sentence: str, processed_labels: List[Dict], has_modifications: bool) -> Dict[str, Any]:
        """
        使用LLM进行质量评估和最终决策

        Args:
            sentence: 原始句子
            processed_labels: 处理后的标签列表
            has_modifications: 是否有修改

        Returns:
            决策结果字典
        """
        # 准备标签信息
        labels_info = []
        for label in processed_labels:
            labels_info.append({
                "aspect": label.get("aspect", ""),
                "polarity": label.get("polarity", ""),
                "standardization_status": label.get("standardization_status", "unchanged"),
                "sentiment_status": label.get("sentiment_status", "unchanged")
            })

        user_prompt = f"""Please evaluate the quality of the following data and make a save decision:

Original sentence: {sentence}

Processed labels: {json.dumps(labels_info, ensure_ascii=False, indent=2)}

Has modifications: {has_modifications}

Please perform sentence optimization (if needed) and quality assessment according to the requirements above, and decide whether to save this data.

Please strictly output results in JSON format."""

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = self.client.chat_completion(
                messages=messages,
                max_tokens=400,
                temperature=0.1
            )

            # 清理响应，移除markdown代码块标记
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # 解析JSON响应
            result = json.loads(cleaned_response)

            # 确保包含所有必要字段
            default_result = {
                "optimized_sentence": sentence,
                "sentence_optimized": False,
                "decision": "discard",
                "quality_score": 1,
                "reasoning": "Unable to parse model response"
            }

            for key in default_result:
                if key not in result:
                    result[key] = default_result[key]

            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}, 原始响应: {response}")
            return {
                "optimized_sentence": sentence,
                "sentence_optimized": False,
                "decision": "discard",
                "quality_score": 1,
                "reasoning": f"JSON parsing error: {str(e)}"
            }
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            return {
                "optimized_sentence": sentence,
                "sentence_optimized": False,
                "decision": "discard",
                "quality_score": 1,
                "reasoning": f"Model call error: {str(e)}"
            }

    def save_single_entry_realtime(self, result: Dict[str, Any], output_path: str):
        """
        实时保存单个处理完成的数据条目
        如果决策为save，立即追加到文件中

        Args:
            result: 单个处理结果
            output_path: 输出文件路径
        """
        if result.get("Decision") != "save":
            return False

        try:
            # 准备单个数据条目
            single_entry = {
                "ID": result["ID"],
                "Sentence": result["OptimizedSentence"],
                "Label": result["FinalLabels"]
            }

            # 每次都检查文件状态，确保模型重新初始化后也能正确工作
            if output_path not in self._incremental_file_state:
                self.initialize_incremental_file(output_path)

            file_state = self._incremental_file_state.get(output_path)

            if file_state is None or not file_state.get('created', False):
                # 第一次写入，创建新文件
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('[\n')
                    json_str = json.dumps(single_entry, ensure_ascii=False, indent=2)
                    f.write(json_str)
                    f.write('\n]')

                # 记录文件状态：已创建且非空
                self._incremental_file_state[output_path] = {
                    'created': True,
                    'has_entries': True,
                    'entry_count': 1
                }

                self.logger.info(f"创建新文件并保存第一条数据: {result['ID']} → {output_path}")

            else:
                # 文件已存在，使用更简单的方法：读取-修改-重写
                try:
                    # 尝试读取现有数据
                    with open(output_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)

                    # 确保是列表
                    if not isinstance(existing_data, list):
                        existing_data = []

                    # 添加新条目
                    existing_data.append(single_entry)

                    # 重写整个文件
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=2)

                    # 更新状态
                    self._incremental_file_state[output_path]['entry_count'] = len(existing_data)
                    self._incremental_file_state[output_path]['has_entries'] = True

                    self.logger.info(f"追加数据到文件: {result['ID']} → {output_path} (总计: {len(existing_data)} 条)")

                except (json.JSONDecodeError, Exception) as e:
                    # 如果读取失败，创建新文件
                    self.logger.warning(f"读取现有文件失败，重新创建: {e}")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump([single_entry], f, ensure_ascii=False, indent=2)

                    self._incremental_file_state[output_path]['has_entries'] = True
                    self._incremental_file_state[output_path]['entry_count'] = 1

            return True

        except Exception as e:
            self.logger.error(f"实时保存数据失败 {result['ID']}: {e}")
            return False

    def initialize_incremental_file(self, output_path: str):
        """
        初始化增量文件，如果文件不存在则创建空文件

        Args:
            output_path: 输出文件路径
        """
        if os.path.exists(output_path):
            # 文件已存在，读取状态
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content == '[]':
                        # 空文件
                        self._incremental_file_state[output_path] = {
                            'created': True,
                            'has_entries': False,
                            'entry_count': 0
                        }
                    else:
                        # 有内容的文件，尝试解析并计算条目数
                        try:
                            data = json.loads(content)
                            entry_count = len(data) if isinstance(data, list) else 0
                            self._incremental_file_state[output_path] = {
                                'created': True,
                                'has_entries': entry_count > 0,
                                'entry_count': entry_count
                            }
                            self.logger.info(f"现有文件包含 {entry_count} 条数据: {output_path}")
                        except:
                            # 文件格式有问题，重新创建
                            self._incremental_file_state[output_path] = {
                                'created': True,
                                'has_entries': False,
                                'entry_count': 0
                            }
            except Exception as e:
                self.logger.warning(f"读取现有文件状态失败，将重新创建: {e}")
                self._incremental_file_state[output_path] = {
                    'created': True,
                    'has_entries': False,
                    'entry_count': 0
                }
        else:
            # 文件不存在，创建空文件
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('[]')

                self._incremental_file_state[output_path] = {
                    'created': True,
                    'has_entries': False,
                    'entry_count': 0
                }

                self.logger.info(f"创建空的增量文件: {output_path}")
            except Exception as e:
                self.logger.error(f"创建增量文件失败: {e}")
                self._incremental_file_state[output_path] = {
                    'created': False,
                    'has_entries': False,
                    'entry_count': 0
                }

    def process_batch_with_realtime_save(self, data_entries: List[Dict[str, Any]],
                                       output_path: str,
                                       delay: float = 1.0) -> Dict[str, Any]:
        """
        批量处理数据条目，支持实时保存
        每处理完一条数据，如果决策为save，立即追加到文件中

        Args:
            data_entries: 模型B输出的数据条目列表
            output_path: 输出文件路径
            delay: 请求之间的延迟（秒）

        Returns:
            处理统计信息
        """
        total = len(data_entries)
        saved_count = 0
        discarded_count = 0
        error_count = 0

        self.logger.info(f"模型C开始处理 {total} 个数据条目（实时保存模式）")

        # 初始化增量文件
        self.initialize_incremental_file(output_path)

        for i, entry in enumerate(data_entries):
            try:
                result = self.process_single_entry(entry)

                # 实时保存
                if result.get("Decision") == "save":
                    if self.save_single_entry_realtime(result, output_path):
                        saved_count += 1
                    else:
                        self.logger.warning(f"数据决策为save但保存失败: {result.get('ID', 'unknown')}")
                        error_count += 1
                else:
                    discarded_count += 1

                # 显示进度
                if (i + 1) % 10 == 0:
                    self.logger.info(f"模型C已处理 {i + 1}/{total} 个条目 (保存: {saved_count}, 丢弃: {discarded_count}, 错误: {error_count})")

                # 添加延迟避免API限制
                if i < total - 1:
                    time.sleep(delay)

            except Exception as e:
                error_count += 1
                self.logger.error(f"处理条目 {entry.get('ID', 'unknown')} 时发生错误: {e}")

        self.logger.info(f"模型C处理完成！总计: {total}, 保存: {saved_count}, 丢弃: {discarded_count}, 错误: {error_count}")

        # 返回统计信息
        return {
            'total_processed': total,
            'saved_count': saved_count,
            'discarded_count': discarded_count,
            'error_count': error_count,
            'save_rate': saved_count / total if total > 0 else 0,
            'output_path': output_path
        }

    def process_batch(self, data_entries: List[Dict[str, Any]],
                     delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        批量处理数据条目（原有方法，保持向后兼容）

        Args:
            data_entries: 模型B输出的数据条目列表
            delay: 请求之间的延迟（秒）

        Returns:
            处理后的数据列表
        """
        results = []
        total = len(data_entries)

        self.logger.info(f"模型C开始处理 {total} 个数据条目")

        for i, entry in enumerate(data_entries):
            result = self.process_single_entry(entry)
            results.append(result)

            # 显示进度
            if (i + 1) % 10 == 0:
                self.logger.info(f"模型C已处理 {i + 1}/{total} 个条目")

            # 添加延迟避免API限制
            if i < total - 1:
                time.sleep(delay)

        self.logger.info(f"模型C处理完成，共处理 {len(results)} 个条目")

        # 统计保存/丢弃情况
        saved_count = sum(1 for r in results if r.get("Decision") == "save")
        discarded_count = len(results) - saved_count
        self.logger.info(f"最终决策: 保存 {saved_count} 条, 丢弃 {discarded_count} 条")

        return results

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        保存处理结果

        Args:
            results: 处理结果列表
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"模型C结果已保存到: {output_path}")
        except Exception as e:
            self.logger.error(f"模型C保存结果时出错: {e}")

    def save_final_dataset(self, results: List[Dict[str, Any]], output_path: str):
        """
        保存最终的数据集（仅保存决策为save的数据）- 增量写入模式（内存友好）

        Args:
            results: 处理结果列表
            output_path: 输出文件路径
        """
        try:
            saved_count = 0

            # 使用增量写入模式，避免在内存中累积大量数据
            with open(output_path, 'w', encoding='utf-8') as f:
                # 写入JSON数组的开始标记
                f.write('[\n')

                first_entry = True
                for result in results:
                    if result.get("Decision") == "save":
                        # 准备单个数据条目
                        single_entry = {
                            "ID": result["ID"],
                            "Sentence": result["OptimizedSentence"],
                            "Label": result["FinalLabels"]
                        }

                        # 如果不是第一个条目，添加逗号分隔符
                        if not first_entry:
                            f.write(',\n')

                        # 将单个条目格式化并写入文件
                        json_str = json.dumps(single_entry, ensure_ascii=False, indent=2)

                        # 处理缩进，保持JSON数组格式一致
                        lines = json_str.split('\n')
                        for i, line in enumerate(lines):
                            if i == 0:
                                f.write(line)  # 第一行不缩进
                            else:
                                f.write('\n  ' + line)  # 后续行缩进2个空格

                        saved_count += 1
                        first_entry = False

                # 写入JSON数组的结束标记
                f.write('\n]')

            self.logger.info(f"最终高质量数据集已保存到: {output_path}")
            self.logger.info(f"高质量数据集包含 {saved_count} 条数据")

        except Exception as e:
            self.logger.error(f"保存最终数据集时出错: {e}")

    def load_model_b_output(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载模型B的输出数据

        Args:
            file_path: 模型B输出文件路径

        Returns:
            数据条目列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"模型C已加载 {len(data)} 个数据条目从: {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"模型C加载数据时出错: {e}")
            return []

if __name__ == "__main__":
    # 示例用法
    model = QualityDecisionModel()

    # 示例数据（模型B的输出）
    sample_data = [
        {
            "ID": "001",
            "Sentence": "The restaurant's prices are very reasonable and the service quality is also excellent",
            "ProcessedLabels": [
                {
                    "aspect": "price",
                    "polarity": "positive",
                    "standardization_status": "modified",
                    "sentiment_status": "unchanged"
                },
                {
                    "aspect": "service quality",
                    "polarity": "positive",
                    "standardization_status": "unchanged",
                    "sentiment_status": "unchanged"
                }
            ],
            "OriginalLabels": [["price", "positive"], ["service quality", "positive"]]
        }
    ]

    # 处理数据
    results = model.process_batch(sample_data)

    # 保存详细结果
    model.save_results(results, "evaluate/model_c_output.json")

    # 保存最终数据集
    model.save_final_dataset(results, "evaluate/final_dataset.json")