#!/usr/bin/env bash
# ============================================================================
# DMR-ABSA 完整流水线
#   阶段 1: 生成 prompt        (utils_prompt/)
#   阶段 2: 解析 LLM 输出       (scripts/process_absa_data.py)
#   阶段 3: PRUS A→B→C 质量评估  (evaluation/)
#   阶段 4: RAG 语义去重        (dual-constrained-dedup/)
#
# 用法:
#   bash run.sh <domain> <shot>     # 例: bash run.sh res 5
#
# 前置条件:
#   1. pip install -r requirements.txt
#   2. 配置环境变量（参见 .env.example）:
#      export OPENAI_API_KEY=sk-xxx
#      export OPENAI_API_BASE=https://api.openai.com/v1
#      export OPENAI_Model=DeepSeek-V3
# ============================================================================
set -euo pipefail

# 自动切换到仓库根目录（脚本所在目录的上一级），
# 因为各阶段均使用仓库相对路径（prompts/、*_driven/ 等）
cd "$(dirname "$0")"

DOMAIN=${1:?用法: bash run.sh <domain: res|lap|res15|res16> <shot: 2|5>}
SHOT=${2:?用法: bash run.sh <domain: res|lap|res15|res16> <shot: 2|5>}
SEED=42
NUM_SAMPLES=4000
NUM_EXAMPLES=4

echo "============================================================"
echo " DMR-ABSA 流水线: domain=${DOMAIN}, ${SHOT}%-shot"
echo "============================================================"

# ----------------------------------------------------------------------------
# 阶段 1a: 关键点驱动（keypoint-driven）prompt 生成
#   输出: prompts/keypoint_prompts/${DOMAIN}_sample${SHOT}.json
# ----------------------------------------------------------------------------
python utils_prompt/get_keypoint_prompts.py \
    --domain "${DOMAIN}" --dataset sample \
    --seed "${SEED}" --num_shots "${SHOT}" \
    --num_samples "${NUM_SAMPLES}" --num_examples "${NUM_EXAMPLES}"

# ----------------------------------------------------------------------------
# 阶段 1b: 实例驱动（instance-driven）prompt 生成
#   输出: prompts/instance_prompts/${DOMAIN}_sample${SHOT}.json
# ----------------------------------------------------------------------------
python utils_prompt/get_instance_prompts.py \
    --domain "${DOMAIN}" --dataset sample \
    --seed "${SEED}" --num_shots "${SHOT}"

# ----------------------------------------------------------------------------
# 阶段 2: 解析 prompt 文件中 LLM 返回的句子与标签
#   keypoint → property-driven/<domain>/
#   instance → seed-driven/<domain>/
# （prompt_gen 字段中的 llm_output 需已由你的 LLM 调用脚本填充）
# ----------------------------------------------------------------------------
python scripts/process_absa_data.py \
    "prompts/keypoint_prompts/${DOMAIN}_sample${SHOT}.json" \
    property-driven "${DOMAIN}_sample${SHOT}" --prompt_type keypoint

python scripts/process_absa_data.py \
    "prompts/instance_prompts/${DOMAIN}_sample${SHOT}.json" \
    seed-driven "${DOMAIN}_sample${SHOT}" --prompt_type instance

# ----------------------------------------------------------------------------
# 阶段 3: PRUS A→B→C 质量评估（方面词标准化 → 情感极性评估 → 质量决策）
#   对应论文中 PRUS 的 3 次精炼迭代（A/B/C 三个智能体阶段）
#   流式逐条处理, 实时落盘, 输出: evaluation/optimized_outputs/
# ----------------------------------------------------------------------------
python evaluation/optimized_streaming.py \
    "property-driven/${DOMAIN}/direct_${DOMAIN}_sample${SHOT}.json"

# ----------------------------------------------------------------------------
# 阶段 4: RAG 语义去重
#   先在 dual-constrained-dedup/config.py 中确认 INPUT_DIR / OUTPUT_DIR
#   输出: dual-constrained-dedup/rag_outputs/*_RAG.json
# ----------------------------------------------------------------------------
python dual-constrained-dedup/rag_retrieval.py

echo "============================================================"
echo " 流水线完成。各阶段输出:"
echo "   prompts/keypoint_prompts/  prompts/instance_prompts/"
echo "   property-driven/       seed-driven/"
echo "   evaluation/optimized_outputs/"
echo "   dual-constrained-dedup/rag_outputs/"
echo "============================================================"
