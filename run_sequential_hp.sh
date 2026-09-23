#!/bin/bash
# Full sequential Algorithm 1 (no max_samples), then WILD on the edited model.
# Args: <data_type> <cap> [lr] [soft_lambda]
set -euo pipefail
cd /home/zwz/01_safeedit

DATA_TYPE=${1:?usage: run_sequential_hp.sh <counterfact|zsre|wiki> <cap> [lr] [soft_lambda]}
CAP=${2:?usage: run_sequential_hp.sh <counterfact|zsre|wiki> <cap> [lr] [soft_lambda]}
LR=${3:-3e-5}
LAM=${4:-0.1}
TRAIN_PY=/home/zwz/miniconda3/envs/sf/bin/python
EVAL_PY=/home/zwz/miniconda3/envs/sf_vllm/bin/python

COMMON=(
  --model llama3-8b
  --data_type "${DATA_TYPE}"
  --sequential_edit
  --num_edits 100
  --batch_size 32
  --edit_cache_style sequential
  --cache_sample_num "${CAP}"
  --lr "${LR}"
  --newton_damping 1e-4
  --soft_lambda "${LAM}"
  --disable_old_loss_check
  --wandb_project RhoEdit
)

echo "[sequential] train ${DATA_TYPE} cap=${CAP} lr=${LR} λ=${LAM} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
TMP=$(mktemp)
PYTHONUNBUFFERED=1 "${TRAIN_PY}" run_rhoedit.py "${COMMON[@]}" | tee "${TMP}"
NAME=$(grep 'Model will be saved to BASE_DIR/' "${TMP}" | tail -1 | sed 's/.*BASE_DIR\///' | tr -d '\r')
rm -f "${TMP}"
if [[ -z "${NAME}" ]]; then
  echo "[sequential] failed to parse saved model name" >&2
  exit 1
fi
if [[ "${NAME}" == *firstchunk* ]]; then
  echo "[sequential] refused first-chunk save name: ${NAME}" >&2
  exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  FIRST_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
  export CUDA_VISIBLE_DEVICES="${FIRST_GPU}"
fi
echo "[sequential] eval ${NAME} on GPU ${CUDA_VISIBLE_DEVICES:-0}"

for CTX in qa_inst no_context; do
  "${EVAL_PY}" run_edited_benchmarks.py \
    --edited_model_dir "${NAME}" \
    --data_type "${DATA_TYPE}" \
    --context_type "${CTX}" \
    --alg_name ft_edit \
    --model_name llama3-8b \
    --evaluation_criteria llm_judge \
    --judge_batch_size 16 \
    --no_wandb
done

echo "[sequential] done ${DATA_TYPE} cap=${CAP} lr=${LR} λ=${LAM}"
