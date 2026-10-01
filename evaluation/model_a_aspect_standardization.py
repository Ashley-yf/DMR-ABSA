"""
模型A：方面词标准化模型
角色：数据清洗专家，专注于方面词标准化和改写处理
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

class AspectStandardizationModel:
    def __init__(self):
        """
        初始化方面词标准化模型
        """
        self.client = create_qnaigc_client()

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.system_prompt = """You are a data cleaning expert specializing in aspect term standardization for sentiment analysis. Given a data entry containing id, sentence, and label (aspect terms and sentiment polarity), your task is to:

1. Carefully analyze the sentence and check if each aspect term from the label appears in the sentence.

2. CRITICAL PRINCIPLES - MUST FOLLOW:
   - ASPECT TERM MUST EXIST IN THE SENTENCE: Every aspect term in final output must appear in the sentence text
   - NO DUPLICATE ASPECTS: Each aspect term must be unique, never repeat the same aspect term
   - MINIMAL MODIFICATION: Prefer keeping original aspect terms, only modify when absolutely necessary
   - NO REWRITING: Do not rewrite or create new aspect terms that aren't explicitly mentioned

3. If the aspect term appears in the sentence (exactly matching), keep it unchanged and set standardization_status to "unchanged".

4. If the aspect term does NOT appear in the sentence:
   - Find aspect term that ACTUALLY EXISTS in the sentence and matches the sentiment
   - NEVER create aspect terms that don't exist in the sentence
   - MUST check against all other aspect terms to avoid duplicates
   - Set standardization_status to "modified"
   - If no suitable aspect exists in sentence, use "REMOVE_ASPECT"

5. DUPLICATE PREVENTION - ABSOLUTE REQUIREMENT:
   - Check ALL aspect terms in the original label
   - NEVER create duplicate aspect terms in final result
   - Each aspect term must be unique
   - If replacement would create duplicate, use "REMOVE_ASPECT" instead

6. Specific matching logic:
   - If original aspect is "price": Keep "price" or related pricing terms if sentence mentions cost/expensive/cheap/affordable
   - If original aspect is "performance": Use the actual item being described (what runs/operates), not abstract "performance"
   - If original aspect is "service": Use staff/attitude related terms if sentence mentions people/interaction

6. Only use "REMOVE_ASPECT" if the sentence contains no meaningful aspect terms at all.

Examples:
- Sentence: "The phone is expensive but the screen is very clear"
  Original aspect: "price" → "price" (unchanged, "expensive" describes price, not phone)
  Original aspect: "screen" → "screen" (unchanged, screen is in sentence)
- Sentence: "Good performance but expensive price"
  Original aspect: "performance" → "performance" (unchanged, "performance" exists in sentence)
  Original aspect: "price" → "price" (unchanged, "price" concept exists in sentence via "expensive")
- Sentence: "The screen is bright but the battery dies quickly"
  Original aspect: "screen" → "screen" (unchanged, exists in sentence)
  Original aspect: "battery" → "battery" (unchanged, exists in sentence)
- Sentence: "Expensive device with poor user experience"
  Original aspect: "price" → "device" (modified, "device" exists in sentence and is described as expensive)
  Original aspect: "experience" → "experience" (unchanged, exists in sentence)
- Sentence: "The food was delicious and service was excellent"
  Original aspect: "food" → "food" (unchanged)
  Original aspect: "service" → "service" (unchanged)
- Sentence: "The laptop runs fast and looks great"
  Original aspect: "performance" → "laptop" (modified, laptop is what runs fast)
- Sentence: "This car has amazing speed but terrible fuel efficiency"
  Original aspect: "speed" → "car" (modified, but careful: "fuel efficiency" might be better)

Please strictly output in the following JSON format:
{
    "standardized_aspect": "exact term from sentence OR REMOVE_ASPECT",
    "standardization_status": "modified/unchanged",
    "reasoning": "detailed analysis of the choice and why it matches the sentiment"
}"""

    def process_single_entry(self, data_entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个数据条目

        Args:
            data_entry: 包含ID, Sentence, Label的数据字典

        Returns:
            处理后的数据字典，包含标准化状态
        """
        try:
            # 提取数据
            entry_id = data_entry.get("ID", "")
            sentence = data_entry.get("Sentence", "")
            labels = data_entry.get("Label", [])

            results = []

            # 处理每个标签
            for label_pair in labels:
                if len(label_pair) >= 2:
                    aspect = label_pair[0]
                    polarity = label_pair[1]

                    # 调用模型进行标准化
                    standardized_aspect, status = self._standardize_aspect(
                        sentence, aspect, polarity
                    )

                    # 只有当方面词没有被移除时才添加到结果中
                    if standardized_aspect is not None:
                        results.append({
                            "aspect": standardized_aspect,
                            "polarity": polarity,
                            "standardization_status": status
                        })

            return {
                "ID": entry_id,
                "Sentence": sentence,
                "ProcessedLabels": results,
                "OriginalLabels": labels
            }

        except Exception as e:
            self.logger.error(f"处理数据条目时出错: {e}")
            return {
                "ID": data_entry.get("ID", ""),
                "Sentence": data_entry.get("Sentence", ""),
                "ProcessedLabels": [],
                "OriginalLabels": data_entry.get("Label", []),
                "error": str(e)
            }

    def _standardize_aspect(self, sentence: str, aspect: str, polarity: str) -> Tuple[str, str]:
        """
        使用LLM标准化方面词

        Args:
            sentence: 原始句子
            aspect: 原始方面词
            polarity: 情感极性

        Returns:
            (标准化后的方面词, 标准化状态)
        """
        user_prompt = f"""Please analyze the following sentence and standardize the aspect term:

Sentence: {sentence}

Original aspect term: {aspect}
Sentiment polarity: {polarity}

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
            standardized_aspect = result.get("standardized_aspect")
            status = result.get("standardization_status", result.get("status", "unchanged"))

            # 特殊处理：如果方面词被标记为需要移除
            if standardized_aspect == "REMOVE_ASPECT":
                return None, "removed"
            else:
                return standardized_aspect, status

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}, 原始响应: {response}")
            return aspect, "unchanged"
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            # 如果LLM调用失败，返回原始方面词
            return aspect, "unchanged"

    def process_batch(self, data_entries: List[Dict[str, Any]],
                     delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        批量处理数据条目

        Args:
            data_entries: 数据条目列表
            delay: 请求之间的延迟（秒）

        Returns:
            处理后的数据列表
        """
        results = []
        total = len(data_entries)

        self.logger.info(f"模型A开始处理 {total} 个数据条目")

        for i, entry in enumerate(data_entries):
            result = self.process_single_entry(entry)
            results.append(result)

            # 显示进度
            if (i + 1) % 10 == 0:
                self.logger.info(f"模型A已处理 {i + 1}/{total} 个条目")

            # 添加延迟避免API限制
            if i < total - 1:
                time.sleep(delay)

        self.logger.info(f"模型A处理完成，共处理 {len(results)} 个条目")
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

            self.logger.info(f"模型A结果已保存到: {output_path}")
            self.logger.info(f"模型A处理了 {len(results)} 个数据条目")
        except Exception as e:
            self.logger.error(f"模型A保存结果时出错: {e}")

    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载原始数据

        Args:
            file_path: 数据文件路径

        Returns:
            数据条目列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"模型A已加载 {len(data)} 个数据条目从: {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"模型A加载数据时出错: {e}")
            return []

if __name__ == "__main__":
    # 示例用法
    model = AspectStandardizationModel()

    # 示例数据
    sample_data = [
        {
            "ID": "001",
            "Sentence": "The restaurant's prices are very reasonable and the service quality is also excellent",
            "Label": [["price", "positive"], ["service quality", "positive"]]
        }
    ]

    # 处理数据
    results = model.process_batch(sample_data)

    # 保存结果
    model.save_results(results, "evaluate/model_a_output.json")