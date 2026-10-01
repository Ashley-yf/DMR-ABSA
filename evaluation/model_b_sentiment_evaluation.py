"""
模型B：情感极性评估模型
角色：情感分析专家，专注于评估和纠正情感极性
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

class SentimentEvaluationModel:
    def __init__(self):
        """
        初始化情感极性评估模型
        """
        self.client = create_qnaigc_client()

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.system_prompt = """You are a sentiment analysis expert specializing in evaluating sentiment polarity of aspect terms. Given a data entry containing id, sentence, and label (aspect terms with standardization_status and sentiment polarity), your task is to:

1. Pay SPECIAL ATTENTION to aspect terms where standardization_status is "modified" - these have been replaced and need sentiment re-evaluation.

2. For each aspect term:
   - If standardization_status is "modified": Carefully re-evaluate the sentiment polarity, as the original polarity was based on a different aspect
   - If standardization_status is "unchanged": Verify the sentiment polarity is still accurate

3. Analyze the sentiment polarity (positive, negative, neutral) in the context of the sentence. Consider:
   - Direct sentiment words (excellent, terrible, expensive, good)
   - Transition words (but, however, although)
   - Modifiers (very, quite, slightly)
   - Context clues

4. Examples:
   - Sentence: "The phone is expensive but the screen is very clear"
     Original: "price"/"positive" → Modified aspect: "phone" → Should be "negative" because "expensive" indicates negative sentiment about price
   - Sentence: "The food was delicious but service was terrible"
     Both aspects unchanged → "food"/"positive", "service"/"negative" (unchanged)
   - Sentence: "This restaurant has great food, terrible service, amazing ambiance"
     Original: "food"/"negative", "service"/"negative", "ambiance"/"negative"
     All aspects should be re-evaluated based on actual context

5. Important sentiment rules:
   - "expensive", "costly", "high-priced" → usually NEGATIVE (too expensive is bad)
   - "cheap", "affordable", "inexpensive" → usually POSITIVE (good price)
   - "fast", "quick" → usually POSITIVE (good performance)
   - "slow", "laggy" → usually NEGATIVE (bad performance)
   - Consider context: "runs fast" (positive) vs "ages fast" (negative)

5. Output the corrected sentiment polarity and status:
   - "modified": if sentiment polarity was changed
   - "unchanged": if sentiment polarity remains the same

Please strictly output in the following JSON format:
{
    "corrected_polarity": "positive/negative/neutral",
    "sentiment_status": "modified/unchanged",
    "reasoning": "analysis of why this polarity was chosen"
}"""

    def process_single_entry(self, data_entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个数据条目（模型A的输出）

        Args:
            data_entry: 包含ID, Sentence, ProcessedLabels的数据字典

        Returns:
            处理后的数据字典，包含情感评估状态
        """
        try:
            # 提取数据
            entry_id = data_entry.get("ID", "")
            sentence = data_entry.get("Sentence", "")
            processed_labels = data_entry.get("ProcessedLabels", [])

            results = []

            # 处理每个已标准化的标签
            for label_info in processed_labels:
                if isinstance(label_info, dict):
                    aspect = label_info.get("aspect", "")
                    polarity = label_info.get("polarity", "")
                    standardization_status = label_info.get("standardization_status", "unchanged")

                    # 调用模型评估情感极性
                    corrected_polarity, sentiment_status = self._evaluate_sentiment(
                        sentence, aspect, polarity, standardization_status
                    )

                    results.append({
                        "aspect": aspect,
                        "polarity": corrected_polarity,
                        "standardization_status": standardization_status,
                        "sentiment_status": sentiment_status
                    })

            return {
                "ID": entry_id,
                "Sentence": sentence,
                "ProcessedLabels": results,
                "OriginalLabels": data_entry.get("OriginalLabels", [])
            }

        except Exception as e:
            self.logger.error(f"处理数据条目时出错: {e}")
            return {
                "ID": data_entry.get("ID", ""),
                "Sentence": data_entry.get("Sentence", ""),
                "ProcessedLabels": [],
                "OriginalLabels": data_entry.get("OriginalLabels", []),
                "error": str(e)
            }

    def _evaluate_sentiment(self, sentence: str, aspect: str, polarity: str, standardization_status: str) -> Tuple[str, str]:
        """
        使用LLM评估情感极性

        Args:
            sentence: 原始句子
            aspect: 标准化后的方面词
            polarity: 原始情感极性
            standardization_status: 方面词标准化状态 ("modified" 或 "unchanged")

        Returns:
            (纠正后的情感极性, 情感评估状态)
        """
        # 根据标准化状态调整提示词的重点
        if standardization_status == "modified":
            emphasis = "This aspect term was MODIFIED/REPLACED, so the original sentiment polarity was based on a different aspect term. Please carefully re-evaluate the sentiment for this NEW aspect term."
        else:
            emphasis = "This aspect term is unchanged, but please verify the sentiment polarity is still accurate in the context."

        user_prompt = f"""Please analyze the sentiment polarity of the aspect term in the following sentence:

Sentence: {sentence}

Aspect term: {aspect}
Original sentiment polarity: {polarity}
Standardization status: {standardization_status}

{emphasis}

Please carefully analyze the true sentiment tendency expressed toward this aspect term in the sentence and determine whether the original sentiment polarity is accurate.

Please analyze according to the requirements above and output in JSON format."""

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = self.client.chat_completion(
                messages=messages,
                max_tokens=200,
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
            corrected_polarity = result.get("corrected_polarity", polarity)
            sentiment_status = result.get("sentiment_status", result.get("status", "unchanged"))

            return corrected_polarity, sentiment_status

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}, 原始响应: {response}")
            return polarity, "unchanged"
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            # 如果LLM调用失败，返回原始情感极性
            return polarity, "unchanged"

    def process_batch(self, data_entries: List[Dict[str, Any]],
                     delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        批量处理数据条目

        Args:
            data_entries: 模型A输出的数据条目列表
            delay: 请求之间的延迟（秒）

        Returns:
            处理后的数据列表
        """
        results = []
        total = len(data_entries)

        self.logger.info(f"模型B开始处理 {total} 个数据条目")

        for i, entry in enumerate(data_entries):
            result = self.process_single_entry(entry)
            results.append(result)

            # 显示进度
            if (i + 1) % 10 == 0:
                self.logger.info(f"模型B已处理 {i + 1}/{total} 个条目")

            # 添加延迟避免API限制
            if i < total - 1:
                time.sleep(delay)

        self.logger.info(f"模型B处理完成，共处理 {len(results)} 个条目")
        return results

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        保存处理结果 - 增量写入模式（内存友好）

        Args:
            results: 处理结果列表
            output_path: 输出文件路径
        """
        try:
            # 使用增量写入模式，避免在内存中累积大量数据
            with open(output_path, 'w', encoding='utf-8') as f:
                # 写入JSON数组的开始标记
                f.write('[\n')

                first_entry = True
                for result in results:
                    # 如果不是第一个条目，添加逗号分隔符
                    if not first_entry:
                        f.write(',\n')

                    # 将单个条目格式化并写入文件
                    json_str = json.dumps(result, ensure_ascii=False, indent=2)

                    # 处理缩进，保持JSON数组格式一致
                    lines = json_str.split('\n')
                    for i, line in enumerate(lines):
                        if i == 0:
                            f.write(line)  # 第一行不缩进
                        else:
                            f.write('\n  ' + line)  # 后续行缩进2个空格

                    first_entry = False

                # 写入JSON数组的结束标记
                f.write('\n]')

            self.logger.info(f"模型B结果已保存到: {output_path}")
            self.logger.info(f"模型B处理了 {len(results)} 个数据条目")
        except Exception as e:
            self.logger.error(f"模型B保存结果时出错: {e}")

    def load_model_a_output(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载模型A的输出数据

        Args:
            file_path: 模型A输出文件路径

        Returns:
            数据条目列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"模型B已加载 {len(data)} 个数据条目从: {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"模型B加载数据时出错: {e}")
            return []

if __name__ == "__main__":
    # 示例用法
    model = SentimentEvaluationModel()

    # 示例数据（模型A的输出）
    sample_data = [
        {
            "ID": "001",
            "Sentence": "The restaurant's prices are very reasonable and the service quality is also excellent",
            "ProcessedLabels": [
                {
                    "aspect": "price",
                    "polarity": "positive",
                    "standardization_status": "modified"
                },
                {
                    "aspect": "service quality",
                    "polarity": "positive",
                    "standardization_status": "unchanged"
                }
            ],
            "OriginalLabels": [["price", "positive"], ["service quality", "positive"]]
        }
    ]

    # 处理数据
    results = model.process_batch(sample_data)

    # 保存结果
    model.save_results(results, "evaluate/model_b_output.json")