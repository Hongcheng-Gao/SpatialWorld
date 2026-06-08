#!/usr/bin/env bash
# 串联跑 8 个闭源模型（dual_agent / ProcTHOR）
# 上一个跑完后无论成功失败都会继续下一个。
#
# 在项目根目录执行:
#   bash dual_agent/run_closed_8_serial.sh
#
# 可选环境变量:
#   WORKERS=5   并行 worker 数（默认 5）
#   PYTHON=python3  Python 解释器

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
WORKERS="${WORKERS:-10}"
AGENT_DIR="dual_agent"

run_one() {
  local label="$1"
  local csv="$2"
  local config="$3"
  local save_name="$4"
  local csv_path="${AGENT_DIR}/${csv}"
  local config_path="${AGENT_DIR}/${config}"
  local exit_code=0

  echo ""
  echo "============================================================"
  echo "  开始: ${label}  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "  csv:    ${csv_path}"
  echo "  config: ${config_path}"
  echo "  save:   ${save_name}"
  echo "  workers: ${WORKERS}"
  echo "============================================================"

  if [[ ! -f "${csv_path}" ]]; then
    echo "  ⚠️  跳过: CSV 不存在 ${csv_path}"
    return 0
  fi
  if [[ ! -f "${config_path}" ]]; then
    echo "  ⚠️  跳过: config 不存在 ${config_path}"
    return 0
  fi

  "${PYTHON}" "${AGENT_DIR}/run_benchmark.py" \
    --csv "${csv_path}" \
    --config "${config_path}" \
    --save-name "${save_name}" \
    --headless \
    --workers "${WORKERS}" \
    || exit_code=$?

  if [[ "${exit_code}" -ne 0 ]]; then
    echo "  ⚠️  结束: ${label} 退出码 ${exit_code}，继续下一个模型"
  else
    echo "  ✓ 完成: ${label}  $(date '+%Y-%m-%d %H:%M:%S')"
  fi
}

# Gemini-2.5-Pro
run_one "Gemini-2.5-Pro" \
  "experiments/csv/procthor/Spatial-Annotation-procthor-Gemini-2.5-pro.csv" \
  "config_close_Gemini-2.5-pro.yaml" \
  "Gemini-2.5-Pro"

# Gemini-3.1-Pro
run_one "Gemini-3.1-Pro" \
  "experiments/csv/procthor/Spatial-Annotation-procthor-Gemini-3-Pro-Preview.csv" \
  "config_close_Gemini-3.1-Pro-Preview.yaml" \
  "Gemini-3.1-Pro"

# Gemini-3-flash
run_one "Gemini-3-flash" \
  "experiments/csv/procthor/Spatial-Annotation-procthor-Gemini-3-Flash-Preview.csv" \
  "config_close_Gemini-3-Flash-Preview.yaml" \
  "Gemini-3-Flash-Preview"

# GPT-5
run_one "GPT-5" \
  "experiments/csv/procthor/Spatial-Annotation-procthor-gpt-5.csv" \
  "config_close_gpt-5.yaml" \
  "GPT-5"

# qwen3.5
run_one "qwen3.5" \
  "experiments/csv/procthor/Spatial-Annotation-procthor-Qwen3.5-Plus.csv" \
  "config_close_qwen3p5.yaml" \
  "qwen3.5"

# GPT-5.4
run_one "GPT-5.4" \
  "experiments/csv/procthor/Spatial-Annotation-procthor-Gpt-5p4.csv" \
  "config_close_Gpt-5p4.yaml" \
  "GPT-5.4"

# kimi-k2.5
run_one "kimi-k2.5" \
  "experiments/csv/procthor/Spatial-Annotation-procthor-Kimi-k25.csv" \
  "config_close_kimi_k25.yaml" \
  "kimi-k2.5"

# doubao-seed-2.0-lite（勿用 config_close_doubao_2.yaml）
run_one "doubao-seed-2.0-lite" \
  "experiments/csv/procthor/Spatial-Annotation-procthor-Doubao-2.csv" \
  "config_close_doubao_2-codingplan.yaml" \
  "doubao-seed-2.0-lite"

echo ""
echo "全部 8 个模型已串联跑完。  $(date '+%Y-%m-%d %H:%M:%S')"
