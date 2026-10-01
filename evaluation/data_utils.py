"""
数据读取和处理工具函数
"""

import json
import os
import glob
import logging
from typing import List, Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    加载单个JSON文件

    Args:
        file_path: JSON文件路径

    Returns:
        数据条目列表
    """
    try:
        # 尝试多种编码方式
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']
        data = None

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    data = json.load(f)
                logger.info(f"Successfully loaded with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue

        if data is None:
            raise ValueError("Unable to decode file with any supported encoding")

        # 确保数据是列表格式
        if not isinstance(data, list):
            data = [data]

        logger.info(f"已加载 {len(data)} 个数据条目从: {file_path}")
        return data

    except Exception as e:
        logger.error(f"加载文件 {file_path} 时出错: {e}")
        return []

def load_data_from_directories(base_dirs: List[str], subdirs: List[str] = None) -> List[Dict[str, Any]]:
    """
    从多个目录中加载数据文件

    Args:
        base_dirs: 基础目录列表
        subdirs: 子目录名称列表（如 ['lap', 'res', 'res5', 'res16']）

    Returns:
        所有数据条目的合并列表
    """
    all_data = []

    if subdirs is None:
        subdirs = ['lap', 'res', 'res5', 'res16']

    for base_dir in base_dirs:
        for subdir in subdirs:
            search_dir = os.path.join(base_dir, subdir)

            # 查找目录下所有JSON文件
            json_files = glob.glob(os.path.join(search_dir, "*.json"))

            for json_file in json_files:
                logger.info(f"正在加载文件: {json_file}")
                data = load_json_file(json_file)
                all_data.extend(data)

    logger.info(f"总共加载了 {len(all_data)} 个数据条目")
    return all_data

def load_data_from_specific_paths(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    从指定的文件路径列表中加载数据

    Args:
        file_paths: 文件路径列表

    Returns:
        所有数据条目的合并列表
    """
    all_data = []

    for file_path in file_paths:
        if os.path.exists(file_path):
            logger.info(f"正在加载文件: {file_path}")
            data = load_json_file(file_path)
            all_data.extend(data)
        else:
            logger.warning(f"文件不存在: {file_path}")

    logger.info(f"总共加载了 {len(all_data)} 个数据条目")
    return all_data

def discover_data_files(base_dirs: List[str], subdirs: List[str] = None) -> List[str]:
    """
    发现指定目录下的所有JSON文件

    Args:
        base_dirs: 基础目录列表
        subdirs: 子目录名称列表

    Returns:
        发现的JSON文件路径列表
    """
    json_files = []

    if subdirs is None:
        subdirs = ['lap', 'res', 'res5', 'res16']

    for base_dir in base_dirs:
        for subdir in subdirs:
            search_dir = os.path.join(base_dir, subdir)

            if os.path.exists(search_dir):
                files = glob.glob(os.path.join(search_dir, "*.json"))
                json_files.extend(files)
                logger.info(f"在 {search_dir} 中发现 {len(files)} 个JSON文件")
            else:
                logger.warning(f"目录不存在: {search_dir}")

    logger.info(f"总共发现 {len(json_files)} 个JSON文件")
    return json_files

def validate_data_structure(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    验证数据结构的完整性

    Args:
        data: 数据条目列表

    Returns:
        验证结果字典
    """
    valid_count = 0
    invalid_count = 0
    issues = []

    # 支持两种字段名格式：首字母大写和小写
    required_fields_sets = [
        ['ID', 'Sentence', 'Label'],  # 首字母大写格式
        ['ID', 'sentence', 'label']    # 小写格式
    ]

    for i, entry in enumerate(data):
        # 检查必需字段（支持任一格式）
        valid_format = None
        for fields in required_fields_sets:
            if all(field in entry for field in fields):
                valid_format = fields
                break

        if not valid_format:
            invalid_count += 1
            issues.append(f"条目 {i}: 缺少必需字段，期望格式1: {required_fields_sets[0]} 或格式2: {required_fields_sets[1]}")
            continue

        # 标准化字段名
        id_field, sentence_field, label_field = valid_format
        if 'sentence' in valid_format:
            # 转换为标准格式（首字母大写）
            entry['Sentence'] = entry.pop('sentence')
            entry['Label'] = entry.pop('label')

        # 检查ID
        if not entry['ID']:
            invalid_count += 1
            issues.append(f"条目 {i}: ID为空")
            continue

        # 检查Sentence
        if not entry['Sentence']:
            invalid_count += 1
            issues.append(f"条目 {i}: Sentence为空")
            continue

        # 检查Label
        labels = entry['Label']
        if not isinstance(labels, list) or len(labels) == 0:
            invalid_count += 1
            issues.append(f"条目 {i}: Label格式错误或为空")
            continue

        # 检查每个标签的格式
        valid_labels = True
        for j, label in enumerate(labels):
            if not isinstance(label, list) or len(label) < 2:
                valid_labels = False
                issues.append(f"条目 {i}, 标签 {j}: 标签格式错误，应为 [aspect, polarity]")
                break

        if valid_labels:
            valid_count += 1
        else:
            invalid_count += 1

    logger.info(f"数据验证完成: 有效 {valid_count} 条, 无效 {invalid_count} 条")

    return {
        "total_count": len(data),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "issues": issues[:10]  # 只返回前10个问题
    }

def save_data(data: List[Dict[str, Any]], output_path: str, indent: int = 2):
    """
    保存数据到JSON文件 - 增量写入模式（内存友好）

    Args:
        data: 要保存的数据列表
        output_path: 输出文件路径
        indent: JSON缩进级别
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 使用增量写入模式，避免在内存中累积大量数据
        with open(output_path, 'w', encoding='utf-8') as f:
            # 写入JSON数组的开始标记
            f.write('[\n')

            first_entry = True
            for entry in data:
                # 如果不是第一个条目，添加逗号分隔符
                if not first_entry:
                    f.write(',\n')

                # 将单个条目格式化并写入文件
                json_str = json.dumps(entry, ensure_ascii=False, indent=indent)

                # 处理缩进，保持JSON数组格式一致
                lines = json_str.split('\n')
                for i, line in enumerate(lines):
                    if i == 0:
                        f.write(line)  # 第一行不缩进
                    else:
                        # 根据indent参数确定缩进空格数
                        indent_spaces = ' ' * indent
                        f.write('\n' + indent_spaces + line)  # 后续行缩进

                first_entry = False

            # 写入JSON数组的结束标记
            f.write('\n]')

        logger.info(f"数据已保存到: {output_path} (增量写入模式，共 {len(data)} 个条目)")

    except Exception as e:
        logger.error(f"保存数据到 {output_path} 时出错: {e}")

def create_data_splits(data: List[Dict[str, Any]],
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      random_seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """
    将数据分割为训练、验证和测试集

    Args:
        data: 原始数据列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子

    Returns:
        包含分割后数据的字典
    """
    import random

    # 验证比例
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("训练、验证和测试集比例之和必须为1")

    # 设置随机种子
    random.seed(random_seed)

    # 打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # 计算分割点
    total_count = len(shuffled_data)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)

    # 分割数据
    train_data = shuffled_data[:train_count]
    val_data = shuffled_data[train_count:train_count + val_count]
    test_data = shuffled_data[train_count + val_count:]

    logger.info(f"数据分割完成: 训练集 {len(train_data)} 条, 验证集 {len(val_data)} 条, 测试集 {len(test_data)} 条")

    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }

def get_data_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    获取数据的基本统计信息

    Args:
        data: 数据列表

    Returns:
        统计信息字典
    """
    if not data:
        return {"error": "数据为空"}

    # 基本统计
    total_entries = len(data)
    total_labels = sum(len(entry.get('Label', [])) for entry in data)
    avg_labels_per_entry = total_labels / total_entries if total_entries > 0 else 0

    # 句子长度统计
    sentence_lengths = [len(entry.get('Sentence', '')) for entry in data]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    # 情感极性统计
    polarity_counts = {}
    aspect_counts = {}

    for entry in data:
        labels = entry.get('Label', entry.get('label', []))
        for label in labels:
            if isinstance(label, list) and len(label) >= 2:
                aspect, polarity = label[0], label[1]

                # 处理aspect可能是列表的情况
                if isinstance(aspect, list):
                    aspect = aspect[0] if aspect else 'unknown'

                # 处理polarity可能是列表的情况
                if isinstance(polarity, list):
                    polarity = polarity[0] if polarity else 'unknown'

                # 统计情感极性
                polarity_counts[polarity] = polarity_counts.get(polarity, 0) + 1

                # 统计方面词
                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1

    # 获取最常见的方面词和情感极性
    top_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_entries": total_entries,
        "total_labels": total_labels,
        "avg_labels_per_entry": round(avg_labels_per_entry, 2),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "polarity_distribution": polarity_counts,
        "top_aspects": top_aspects,
        "unique_aspects": len(aspect_counts)
    }

if __name__ == "__main__":
    # 示例用法

    # 定义要搜索的目录
    base_dirs = [
        "property-driven",
        "seed-driven"
    ]

    # 发现所有数据文件
    json_files = discover_data_files(base_dirs)
    print(f"发现的文件: {json_files}")

    # 加载数据
    data = load_data_from_directories(base_dirs)

    # 验证数据结构
    validation_result = validate_data_structure(data)
    print(f"验证结果: {validation_result}")

    # 获取统计信息
    stats = get_data_statistics(data)
    print(f"数据统计: {stats}")