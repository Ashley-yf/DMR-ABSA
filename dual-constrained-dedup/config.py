# -*- coding: utf-8 -*-
"""
RAG模块配置文件
用于配置输入输出路径、处理参数等
"""

import os

# 仓库根目录（dual-constrained-dedup/ 的上一级）
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 输入目录（待去重的评估输出数据）
INPUT_DIR = os.path.join(REPO_ROOT, "evaluation", "outputs")

# 输出目录（去重后数据存储路径，自动创建）
OUTPUT_DIR = os.path.join(REPO_ROOT, "dual-constrained-dedup", "rag_outputs")

# 日志目录（自动创建）
LOG_DIR = os.path.join(REPO_ROOT, "dual-constrained-dedup", "logs")

# 相似度阈值（0-1之间，默认0.9即90%）
SIMILARITY_THRESHOLD = 0.9

# 是否递归遍历子目录（True/False，默认False）
RECURSIVE_TRAVERSAL = False

# 句子嵌入模型（轻量高效，适配短文本；长文本可替换为"all-MiniLM-L12-v2"）
# 既支持本地模型目录路径，也支持 HuggingFace 模型名 "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# 目标处理文件后缀（仅处理指定后缀的文件，默认.json）
TARGET_FILE_SUFFIX = ".json"

# JSON输出缩进（保持与原始格式一致，默认2空格）
JSON_INDENT = 2

# 编码格式（默认utf-8，避免中文乱码）
ENCODING = "utf-8"

# 日志文件名
PROCESSING_LOG_FILE = "processing_log.txt"
SUMMARY_LOG_FILE = "summary_log.txt"

# 调试模式（启用详细日志）
DEBUG_MODE = False

# 批处理大小（避免内存占用过大，可选）
BATCH_SIZE = 1000