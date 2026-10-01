# 优化的流水线处理器使用说明

## 概述

`optimized_streaming.py` 是完全优化的流水线处理器，满足你的所有需求：

### 核心特性

1. **按文件逐个处理** ✅
   - 每个文件独立加载和处理
   - 保持文件边界和数据完整性

2. **单条数据的A→B→C流水线处理** ✅
   - 每条数据依次经过三个模型的处理
   - 完整的流水线：A → B → C

3. **只保存C处理后的最终结果** ✅
   - 模型A和B的中间结果不保存到文件
   - 只有模型C决策为"save"的数据才保存
   - 极致节省资源和内存

4. **ID增量分配机制** ✅
   - 第一条数据ID都为0，之后就ID增量变化
   - 每个文件的ID从0开始重新计数
   - 支持数据去重和追踪

5. **极致内存友好** ✅
   - 处理完一条立即保存，不累积任何中间结果
   - 任何时候只在内存中保留当前处理的单条数据
   - 支持处理超大数据集而不内存溢出

6. **高效的实时统计** ✅
   - 详细的处理进度和时间统计
   - 实时显示保存率、处理速度等关键指标
   - 支持断点续传和恢复

7. **智能的日志系统** ✅
   - 结构化的日志记录，便于调试和监控
   - 区分不同级别的日志信息
   - 详细的错误处理和恢复机制

## 使用方法

### 基本使用

```bash
# 设置API环境变量
export OPENAI_API_KEY=your_api_key_here
export OPENAI_API_BASE=https://llmapi.paratera.com
export OPENAI_Model=DeepSeek-V3

# 运行优化的流水线处理器
python optimized_streaming.py
```

### 高级用法

```bash
# 自定义处理参数
python optimized_streaming.py \
    --base-dirs property-driven seed-driven \
    --subdirs lap res res5 res16 \
    --output-dir custom_outputs \
    --delay 0.5
```

### 与原有实现的对比

| 特性 | 原实现 | 优化实现 | 优势 |
|--------|---------|------------|------|
| 处理模式 | 批量阶段式 | 流式逐个处理 | 内存友好，实时可见 |
| 保存策略 | 混合保存中间结果 | 只保存最终高质量结果 | 节省资源，减少文件大小 |
| 内存使用 | O(n×m) | O(m) | 适合大数据集处理 |
| ID分配 | 简单顺序 | 增量分配 | 支持数据去重和追踪 |
| 实时保存 | 最后统一保存 | 处理完一条立即保存 | 用户体验好，支持断点续传 |

## 核心API集成

支持所有原有的模型：
- `model_a_aspect_standardization.py` - 方面词标准化
- `model_b_sentiment_evaluation.py` - 情感极性评估
- `model_c_quality_decision.py` - 质量决策与优化

## 输出文件命名

格式：`{数据集类型}_{原始文件名}.json`

示例：
- `property-driven/lap/restaurant_reviews.json` → `norm_restaurant_reviews.json`
- `seed-driven/res/movie_reviews.json` → `refined_movie_reviews.json`

## 验证和测试

运行演示验证所有功能：
```bash
python simple_streaming_demo.py
```

这个演示脚本包含：
1. 完整的流水线处理演示
2. 各个模型的独立处理测试
3. 输出格式验证
4. 统计和性能分析

## 性能优化

- **API调用效率**：只调用必要的API，不浪费中间结果保存
- **内存使用**：O(m) 而不是 O(n×m)
- **处理速度**：支持超大数据集处理
- **资源利用**：只保留高质量数据，减少存储成本

这个优化版本完全满足你的需求，并且性能更优、资源使用更高效！