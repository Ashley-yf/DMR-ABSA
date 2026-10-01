"""
数据评估主运行文件
整合三个模型的顺序流处理流程
"""

import os
import time
import logging
import argparse
import sys
from typing import List, Dict, Any

# 导入三个模型和数据工具
from model_a_aspect_standardization import AspectStandardizationModel
from model_b_sentiment_evaluation import SentimentEvaluationModel
from model_c_quality_decision import QualityDecisionModel
from data_utils import (
    load_data_from_directories,
    save_data,
    validate_data_structure,
    get_data_statistics,
    discover_data_files
)
# 导入并发保护模块
from concurrency_protection import (
    ConcurrencyProtector,
    check_running_evaluation_processes
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataEvaluationPipeline:
    """
    数据评估流水线
    整合模型A、B、C的顺序流处理
    """

    def __init__(self, delay: float = 1.0, realtime_save: bool = True):
        """
        初始化评估流水线

        Args:
            delay: 模型调用之间的延迟（秒）
            realtime_save: 是否启用实时保存（处理完一条立即保存）
        """
        self.delay = delay
        self.realtime_save = realtime_save

        # 初始化三个模型
        logger.info("初始化模型A：方面词标准化模型...")
        self.model_a = AspectStandardizationModel()

        logger.info("初始化模型B：情感极性评估模型...")
        self.model_b = SentimentEvaluationModel()

        logger.info("初始化模型C：句子质量提升与最终决策模型...")
        self.model_c = QualityDecisionModel()

        # 创建输出目录
        self.output_dir = "evaluate/outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"实时保存模式: {'启用' if realtime_save else '禁用'}")

    def run_evaluation(self, data: List[Dict[str, Any]],
                      save_intermediate: bool = True,
                      dataset_name: str = "default") -> Dict[str, Any]:
        """
        运行完整的三模型评估流程

        Args:
            data: 原始数据列表
            save_intermediate: 是否保存中间结果
            dataset_name: 数据集名称

        Returns:
            评估结果统计
        """
        start_time = time.time()

        logger.info(f"开始数据评估流程，数据集: {dataset_name}")
        logger.info(f"待处理数据条目数: {len(data)}")

        # 步骤1: 数据验证和统计
        logger.info("=" * 50)
        logger.info("步骤1: 数据验证和统计")
        logger.info("=" * 50)

        validation_result = validate_data_structure(data)
        logger.info(f"数据验证结果: 有效 {validation_result['valid_count']} 条, "
                   f"无效 {validation_result['invalid_count']} 条")

        data_stats = get_data_statistics(data)
        logger.info(f"数据统计: 总条目 {data_stats['total_entries']}, "
                   f"总标签 {data_stats['total_labels']}, "
                   f"平均每条目 {data_stats['avg_labels_per_entry']} 个标签")

        # 步骤2: 模型A - 方面词标准化
        logger.info("=" * 50)
        logger.info("步骤2: 模型A - 方面词标准化")
        logger.info("=" * 50)

        model_a_results = self.model_a.process_batch(data, delay=self.delay)

        # 注释掉中间结果保存 - 只保留最终高质量数据集
        # if save_intermediate:
        #     model_a_output_path = os.path.join(self.output_dir, f"{dataset_name}_model_a_output.json")
        #     self.model_a.save_results(model_a_results, model_a_output_path)

        # 步骤3: 模型B - 情感极性评估
        logger.info("=" * 50)
        logger.info("步骤3: 模型B - 情感极性评估")
        logger.info("=" * 50)

        model_b_results = self.model_b.process_batch(model_a_results, delay=self.delay)

        # 注释掉中间结果保存 - 只保留最终高质量数据集
        # if save_intermediate:
        #     model_b_output_path = os.path.join(self.output_dir, f"{dataset_name}_model_b_output.json")
        #     self.model_b.save_results(model_b_results, model_b_output_path)

        # 步骤4: 模型C - 质量提升与最终决策
        logger.info("=" * 50)
        logger.info("步骤4: 模型C - 质量提升与最终决策")
        logger.info("=" * 50)

        final_dataset_path = os.path.join(self.output_dir, f"{dataset_name}_final_dataset.json")

        if self.realtime_save:
            # 使用实时保存模式
            logger.info("使用实时保存模式 - 处理完一条数据立即保存")
            model_c_stats = self.model_c.process_batch_with_realtime_save(
                model_b_results,
                output_path=final_dataset_path,
                delay=self.delay
            )

            saved_count = model_c_stats['saved_count']
            discarded_count = model_c_stats['discarded_count']
            error_count = model_c_stats['error_count']

        else:
            # 使用原有的批量保存模式
            logger.info("使用批量保存模式 - 处理完所有数据后统一保存")
            model_c_results = self.model_c.process_batch(model_b_results, delay=self.delay)

            # 注释掉model C的中间结果保存
            # final_output_path = os.path.join(self.output_dir, f"{dataset_name}_model_c_output.json")
            # self.model_c.save_results(model_c_results, final_output_path)

            # 保存最终高质量数据集 - 使用原名+final_dataset.json格式
            self.model_c.save_final_dataset(model_c_results, final_dataset_path)

            saved_count = sum(1 for r in model_c_results if r.get("Decision") == "save")
            discarded_count = len(model_c_results) - saved_count
            error_count = 0

        # 计算最终统计
        end_time = time.time()
        total_time = end_time - start_time

        evaluation_stats = {
            "dataset_name": dataset_name,
            "original_count": len(data),
            "valid_count": validation_result['valid_count'],
            "invalid_count": validation_result['invalid_count'],
            "saved_count": saved_count,
            "discarded_count": discarded_count,
            "error_count": error_count if 'error_count' in locals() else 0,
            "save_rate": saved_count / len(data) if data else 0,
            "processing_time_minutes": total_time / 60,
            "data_stats": data_stats,
            "realtime_save": self.realtime_save,
            "outputs": {
                "model_a_output": None,
                "model_b_output": None,
                "model_c_detailed": None,
                "final_dataset": final_dataset_path
            }
        }

        logger.info("=" * 50)
        logger.info("评估完成！统计结果：")
        logger.info("=" * 50)
        logger.info(f"处理模式: {'实时保存' if self.realtime_save else '批量保存'}")
        logger.info(f"原始数据条目: {evaluation_stats['original_count']}")
        logger.info(f"有效数据条目: {evaluation_stats['valid_count']}")
        logger.info(f"最终保存条目: {evaluation_stats['saved_count']}")
        logger.info(f"丢弃条目: {evaluation_stats['discarded_count']}")
        if evaluation_stats.get('error_count', 0) > 0:
            logger.info(f"错误条目: {evaluation_stats['error_count']}")
        logger.info(f"保存率: {evaluation_stats['save_rate']:.2%}")
        logger.info(f"总处理时间: {evaluation_stats['processing_time_minutes']:.2f} 分钟")
        logger.info(f"最终数据集: {final_dataset_path}")

        return evaluation_stats

def main():
    """
    主函数 - 处理命令行参数并运行评估
    """
    parser = argparse.ArgumentParser(description="方面情感分析数据评估工具")

    parser.add_argument(
        "--base-dirs",
        nargs="+",
        default=["property-driven", "seed-driven"],
        help="数据目录列表"
    )

    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=["lap", "res", "res5", "res16"],
        help="子目录列表"
    )

    parser.add_argument(
        "--dataset-name",
        default="combined_dataset",
        help="数据集名称（用于输出文件命名）"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="模型调用之间的延迟（秒）"
    )

    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="不保存中间结果文件"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅验证数据结构，不运行模型评估"
    )

    parser.add_argument(
        "--no-realtime-save",
        action="store_true",
        help="禁用实时保存，使用批量保存模式（默认启用实时保存）"
    )

    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="强制清理无效的锁文件（用于恢复）"
    )

    parser.add_argument(
        "--check-locks",
        action="store_true",
        help="检查是否有评估进程在运行"
    )

    args = parser.parse_args()

    # 处理锁相关操作
    if args.force_clean:
        from concurrency_protection import force_clean_stale_locks
        logger.info("清理无效的锁文件...")
        force_clean_stale_locks()
        return

    if args.check_locks:
        logger.info("检查运行中的评估进程...")
        if check_running_evaluation_processes():
            logger.info("发现运行中的评估进程")
        else:
            logger.info("没有发现运行中的评估进程")
        return

    logger.info("开始数据评估流程...")
    logger.info(f"数据目录: {args.base_dirs}")
    logger.info(f"子目录: {args.subdirs}")
    logger.info(f"数据集名称: {args.dataset_name}")

    # 发现数据文件
    json_files = discover_data_files(args.base_dirs, args.subdirs)
    logger.info(f"发现 {len(json_files)} 个JSON文件")

    if not json_files:
        logger.error("未找到任何数据文件，请检查路径设置")
        return

    # 加载数据
    logger.info("开始加载数据...")
    data = load_data_from_directories(args.base_dirs, args.subdirs)

    if not data:
        logger.error("未能加载任何数据")
        return

    # 如果只是验证模式
    if args.validate_only:
        logger.info("=" * 50)
        logger.info("数据验证模式")
        logger.info("=" * 50)

        validation_result = validate_data_structure(data)
        data_stats = get_data_statistics(data)

        print("\n数据验证结果:")
        print(f"总条目数: {validation_result['total_count']}")
        print(f"有效条目: {validation_result['valid_count']}")
        print(f"无效条目: {validation_result['invalid_count']}")

        if validation_result['issues']:
            print("\n前10个问题:")
            for issue in validation_result['issues']:
                print(f"- {issue}")

        print(f"\n数据统计:")
        print(f"平均每条目标签数: {data_stats['avg_labels_per_entry']}")
        print(f"平均句子长度: {data_stats['avg_sentence_length']}")
        print(f"情感极性分布: {data_stats['polarity_distribution']}")
        print(f"唯一方面词数: {data_stats['unique_aspects']}")

        return

    # 并发保护检查
    logger.info("检查并发访问...")
    if check_running_evaluation_processes():
        logger.error("检测到已有评估进程在运行，为避免数据冲突，当前进程退出")
        sys.exit(1)

    # 运行完整评估流程
    realtime_save = not args.no_realtime_save
    pipeline = DataEvaluationPipeline(delay=args.delay, realtime_save=realtime_save)

    # 使用并发保护器
    with ConcurrencyProtector() as protector:
        if not protector.acquire_process_lock("evaluation"):
            logger.error("无法获取进程锁，可能已有其他评估进程在运行")
            sys.exit(1)

        logger.info("成功获取进程锁，开始评估流程...")

        try:
            evaluation_stats = pipeline.run_evaluation(
                data=data,
                save_intermediate=not args.no_intermediate,
                dataset_name=args.dataset_name
            )

            # 保存评估统计
            stats_path = f"evaluate/outputs/{args.dataset_name}_evaluation_stats.json"
            save_data(evaluation_stats, stats_path)

            logger.info(f"评估统计已保存到: {stats_path}")

        except KeyboardInterrupt:
            logger.info("用户中断了评估流程")
        except Exception as e:
            logger.error(f"评估过程中出现错误: {e}")
            raise

if __name__ == "__main__":
    main()