#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG检索模块 - 独立的文档去重处理系统
功能：基于方面词匹配 + 句子相似度阈值进行精准去重，保持原始格式
作者：RAG Module
"""

import os
import json
import glob
import time
from datetime import datetime
from typing import List, Dict, Set, Tuple, Any
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config import *


class RAGRetrievalModule:
    def __init__(self):
        """初始化RAG检索模块"""
        # 确保输出目录和日志目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

        # 初始化日志
        self.setup_logging()

        # 初始化全局状态
        self.deduplication_cache = []  # 存储{"sentence_embedding": embedding, "aspect_set": set}

        # 初始化句子嵌入模型
        self.logger.info(f"Loading sentence embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.logger.info("Model loaded successfully")

        # 统计信息
        self.stats = {
            'total_files': 0,
            'total_original_items': 0,
            'total_deduped_items': 0,
            'total_duplicates_removed': 0,
            'processing_time': 0,
            'successful_files': 0,
            'failed_files': 0
        }

    def setup_logging(self):
        """设置日志系统"""
        # 创建日志文件路径
        processing_log_path = os.path.join(LOG_DIR, PROCESSING_LOG_FILE)

        # 配置日志系统
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(processing_log_path, encoding=ENCODING),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 50)
        self.logger.info("RAG检索模块启动")
        self.logger.info("=" * 50)

    def extract_aspect_set(self, label: List[List[str]]) -> Set[str]:
        """
        从Label字段提取方面词集合

        Args:
            label: Label字段（二维数组）

        Returns:
            方面词集合
        """
        aspect_set = set()
        try:
            for item in label:
                if len(item) >= 1:
                    aspect_set.add(item[0])
        except Exception as e:
            self.logger.warning(f"Failed to extract aspect set from label: {label}, error: {e}")

        return aspect_set

    def calculate_sentence_embedding(self, sentence: str) -> np.ndarray:
        """
        计算句子的嵌入向量

        Args:
            sentence: 输入句子

        Returns:
            句子嵌入向量
        """
        try:
            return self.embedding_model.encode(sentence, convert_to_tensor=False)
        except Exception as e:
            self.logger.error(f"Failed to calculate embedding for sentence: {sentence}, error: {e}")
            return np.zeros(384)  # 返回零向量（默认维度）

    def is_duplicate(self, sentence: str, aspect_set: Set[str]) -> Tuple[bool, float]:
        """
        判断是否为重复数据（必须同时满足两个条件）
        条件1：方面词集合完全匹配
        条件2：句子相似度≥90%

        Args:
            sentence: 当前句子
            aspect_set: 当前方面词集合

        Returns:
            (是否重复, 最高相似度)
        """
        current_embedding = self.calculate_sentence_embedding(sentence)
        max_similarity = 0.0

        for cached_item in self.deduplication_cache:
            cached_aspect_set = cached_item['aspect_set']

            # 条件1：方面词集合完全匹配
            if aspect_set == cached_aspect_set:
                # 条件2：计算句子相似度
                cached_embedding = cached_item['sentence_embedding']

                try:
                    similarity = cosine_similarity(
                        current_embedding.reshape(1, -1),
                        cached_embedding.reshape(1, -1)
                    )[0][0]

                    max_similarity = max(max_similarity, similarity)

                    # 必须同时满足两个条件才判定为重复
                    if similarity >= SIMILARITY_THRESHOLD:
                        self.logger.debug(f"重复判定: 方面词集合匹配 + 相似度{similarity:.3f}≥{SIMILARITY_THRESHOLD}")
                        return True, similarity
                    else:
                        self.logger.debug(f"非重复: 方面词集合匹配但相似度{similarity:.3f}＜{SIMILARITY_THRESHOLD}")

                except Exception as e:
                    self.logger.warning(f"Failed to calculate similarity, error: {e}")
                    continue
            else:
                self.logger.debug(f"非重复: 方面词集合不匹配")

        return False, max_similarity

    def add_to_cache(self, sentence: str, aspect_set: Set[str]):
        """
        将数据添加到去重缓存

        Args:
            sentence: 句子
            aspect_set: 方面词集合
        """
        embedding = self.calculate_sentence_embedding(sentence)
        self.deduplication_cache.append({
            'sentence_embedding': embedding,
            'aspect_set': aspect_set
        })

    def validate_data_format(self, data: List[Dict[str, Any]]) -> bool:
        """
        验证数据格式是否正确

        Args:
            data: 数据列表

        Returns:
            是否格式正确
        """
        if not isinstance(data, list):
            return False

        for item in data:
            if not isinstance(item, dict):
                return False

            # 检查必要字段
            required_fields = ['ID', 'Sentence', 'Label']
            for field in required_fields:
                if field not in item:
                    return False

            # 检查Label格式（应该是二维数组）
            if not isinstance(item['Label'], list):
                return False

            for label_item in item['Label']:
                if not isinstance(label_item, list) or len(label_item) < 2:
                    return False

        return True

    def process_single_file(self, file_path: str) -> bool:
        """
        处理单个文件

        Args:
            file_path: 文件路径

        Returns:
            是否处理成功
        """
        start_time = time.time()
        file_name = os.path.basename(file_path)
        self.logger.info(f"开始处理文件: {file_name}")

        # 每个文件的本地ID计数器（从0开始）
        local_id_counter = 0

        try:
            # 读取文件
            with open(file_path, 'r', encoding=ENCODING) as f:
                data = json.load(f)

            # 验证数据格式
            if not self.validate_data_format(data):
                self.logger.error(f"文件 {file_name} 数据格式不正确，跳过处理")
                self.stats['failed_files'] += 1
                return False

            original_count = len(data)
            self.stats['total_original_items'] += original_count

            # 处理数据去重
            current_file_valid_data = []

            for item in data:
                try:
                    sentence = item['Sentence']
                    label = item['Label']
                    original_id = item.get('ID', 'unknown')

                    # 提取方面词集合
                    aspect_set = self.extract_aspect_set(label)

                    # 判断是否重复
                    is_dup, similarity = self.is_duplicate(sentence, aspect_set)

                    if is_dup:
                        # 记录重复数据
                        self.logger.info(f"重复数据 - 原ID: {original_id}, 文件: {file_name}, "
                                       f"相似度: {similarity:.3f}≥{SIMILARITY_THRESHOLD}")
                        self.stats['total_duplicates_removed'] += 1
                    else:
                        # 保留数据，重构ID
                        new_item = {
                            'ID': local_id_counter,
                            'Sentence': sentence,
                            'Label': label  # 保持原始格式不变
                        }
                        current_file_valid_data.append(new_item)

                        # 添加到缓存和更新ID计数器
                        self.add_to_cache(sentence, aspect_set)
                        local_id_counter += 1

                except Exception as e:
                    self.logger.warning(f"处理单条数据时出错，文件: {file_name}, 原ID: {item.get('ID', 'unknown')}, 错误: {e}")
                    continue

            # 输出处理后的文件
            output_filename = os.path.splitext(file_name)[0] + '_RAG.json'
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            with open(output_path, 'w', encoding=ENCODING) as f:
                json.dump(current_file_valid_data, f, indent=JSON_INDENT, ensure_ascii=False)

            # 记录处理结果
            processing_time = time.time() - start_time
            deduped_count = len(current_file_valid_data)
            self.stats['total_deduped_items'] += deduped_count

            self.logger.info(f"文件处理完成: {file_name}")
            self.logger.info(f"  - 原始数据条数: {original_count}")
            self.logger.info(f"  - 去重后条数: {deduped_count}")
            self.logger.info(f"  - 重复丢弃条数: {original_count - deduped_count}")
            self.logger.info(f"  - 处理时间: {processing_time:.2f}秒")
            self.logger.info(f"  - 输出文件: {output_path}")

            self.stats['successful_files'] += 1
            return True

        except Exception as e:
            self.logger.error(f"处理文件 {file_name} 时发生错误: {e}")
            self.stats['failed_files'] += 1
            return False

    def get_target_files(self) -> List[str]:
        """
        获取所有目标处理文件

        Returns:
            文件路径列表
        """
        pattern = os.path.join(INPUT_DIR, f"*{TARGET_FILE_SUFFIX}")

        if RECURSIVE_TRAVERSAL:
            pattern = os.path.join(INPUT_DIR, f"**/*{TARGET_FILE_SUFFIX}")
            files = glob.glob(pattern, recursive=True)
        else:
            files = glob.glob(pattern)

        # 按文件名排序
        files.sort()

        self.logger.info(f"找到 {len(files)} 个目标文件")
        for file_path in files:
            self.logger.info(f"  - {os.path.basename(file_path)}")

        return files

    def run(self):
        """运行RAG检索模块"""
        self.logger.info("=" * 50)
        self.logger.info("开始RAG去重处理")
        self.logger.info(f"输入目录: {INPUT_DIR}")
        self.logger.info(f"输出目录: {OUTPUT_DIR}")
        self.logger.info(f"相似度阈值: {SIMILARITY_THRESHOLD}")
        self.logger.info(f"嵌入模型: {EMBEDDING_MODEL}")
        self.logger.info("=" * 50)

        start_time = time.time()

        # 获取目标文件
        target_files = self.get_target_files()
        self.stats['total_files'] = len(target_files)

        if not target_files:
            self.logger.warning("未找到目标文件，处理结束")
            return

        # 逐个处理文件
        for file_path in target_files:
            self.process_single_file(file_path)

        # 计算总处理时间
        self.stats['processing_time'] = time.time() - start_time

        # 生成汇总日志
        self.generate_summary_log()

    def generate_summary_log(self):
        """生成汇总日志"""
        summary_log_path = os.path.join(LOG_DIR, SUMMARY_LOG_FILE)

        with open(summary_log_path, 'w', encoding=ENCODING) as f:
            f.write("RAG检索模块处理汇总报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入目录: {INPUT_DIR}\n")
            f.write(f"输出目录: {OUTPUT_DIR}\n")
            f.write(f"相似度阈值: {SIMILARITY_THRESHOLD}\n")
            f.write(f"嵌入模型: {EMBEDDING_MODEL}\n")
            f.write("\n")
            f.write("处理统计:\n")
            f.write("-" * 30 + "\n")
            f.write(f"总处理文件数: {self.stats['total_files']}\n")
            f.write(f"成功处理文件数: {self.stats['successful_files']}\n")
            f.write(f"失败处理文件数: {self.stats['failed_files']}\n")
            f.write(f"总原始数据条数: {self.stats['total_original_items']}\n")
            f.write(f"总去重后数据条数: {self.stats['total_deduped_items']}\n")
            f.write(f"总重复丢弃条数: {self.stats['total_duplicates_removed']}\n")
            f.write(f"去重率: {self.stats['total_duplicates_removed']/max(self.stats['total_original_items'], 1)*100:.2f}%\n")
            f.write(f"总处理耗时: {self.stats['processing_time']:.2f}秒\n")

        # 同时输出到控制台
        self.logger.info("=" * 50)
        self.logger.info("处理汇总报告")
        self.logger.info("=" * 50)
        self.logger.info(f"总处理文件数: {self.stats['total_files']}")
        self.logger.info(f"成功处理文件数: {self.stats['successful_files']}")
        self.logger.info(f"失败处理文件数: {self.stats['failed_files']}")
        self.logger.info(f"总原始数据条数: {self.stats['total_original_items']}")
        self.logger.info(f"总去重后数据条数: {self.stats['total_deduped_items']}")
        self.logger.info(f"总重复丢弃条数: {self.stats['total_duplicates_removed']}")
        self.logger.info(f"去重率: {self.stats['total_duplicates_removed']/max(self.stats['total_original_items'], 1)*100:.2f}%")
        self.logger.info(f"总处理耗时: {self.stats['processing_time']:.2f}秒")
        self.logger.info("RAG检索模块处理完成！")


def main():
    """主函数"""
    try:
        # 创建并运行RAG检索模块
        rag_module = RAGRetrievalModule()
        rag_module.run()

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()