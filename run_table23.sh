#!/bin/bash
# ICLR'27 RhoEdit Table 2 (n sweep) and Table 3 (λ sweep).
# Defaults match the Llama-3-8B sequential recipe: layers 14-23, lr=5e-5,
# newton_damping=1e-4, λ=10, n=100, 3 GPUs.
#
# Usage on the server, from the repo root (or anywhere; the script cds itself):
#   bash run_table23.sh            # train all cells, wait, then eval
#   bash run_table23.sh train
#   bash run_table23.sh eval
#   bash run_table23.sh collect
#
# Optional env:
#   PY  MODEL  TABLE  TRAIN_GPUS  WANDB
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

PY="${PY:-/home/zwz/miniconda3/envs/sf/bin/python}"
MODEL="${MODEL:-llama3-8b}"
TABLE="${TABLE:-both}"
TRAIN_GPUS="${TRAIN_GPUS:-3}"
MIN_VRAM_GB="${MIN_VRAM_GB:-45}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-72}"
MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-6}"
STAGE="${1:-all}"
LOGDIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOGDIR"

PY_ARGS=(run_table23.py --model "$MODEL" --table "$TABLE" --python "$PY" --train-gpus "$TRAIN_GPUS" --no_wandb)
if [[ "${WANDB:-0}" == "1" ]]; then
    PY_ARGS=(run_table23.py --model "$MODEL" --table "$TABLE" --python "$PY" --train-gpus "$TRAIN_GPUS" --enable_wandb)
fi

MAX_WAIT_SEC=$(echo "$MAX_WAIT_HOURS * 3600" | bc | cut -d. -f1)
START_TIME=$(date +%s)

load_task_list() {
    local action="$1"
    mapfile -t TASK_LIST < <("$PY" "${PY_ARGS[@]}" "$action")
}

get_free_gpus() {
    local min_mb
    min_mb=$(echo "$MIN_VRAM_GB * 1024" | bc | cut -d. -f1)
    local skip="${SKIP_GPU_IDS:-}"
    nvidia-smi --query-gpu=index,memory.free \
               --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v min="$min_mb" -v skip="$skip" '
        BEGIN {
            n = split(skip, a, ",")
            for (i = 1; i <= n; i++) {
                gsub(/ /,"",a[i])
                if (a[i] != "") bad[a[i]] = 1
            }
        }
        {
            id=$1; free=$2
            gsub(/ /,"",id); gsub(/ /,"",free)
            if (free+0 >= min+0 && !(id in bad)) print id
        }
    ' | sort -nr
}

gpu_status_str() {
    nvidia-smi --query-gpu=index,memory.free,memory.total \
               --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{
        id=$1; free=$2; total=$3
        gsub(/ /,"",id); gsub(/ /,"",free); gsub(/ /,"",total)
        printf "  GPU%s:%.0f/%.0fGB", id, free/1024, total/1024
    }'
}

run_stage() {
    local stage_name="$1"
    local total=${#TASK_LIST[@]}
    if [[ $total -eq 0 ]]; then
        echo "[table23] Stage '$stage_name' has no tasks."
        return 0
    fi

    local -a TASK_GPUS=()
    local -a TASK_CMDS=()
    local -a LAUNCHED=()
    local -a TASK_PIDS=()
    local -a TASK_GPU_IDS=()
    local entry n_gpus cmd i
    for entry in "${TASK_LIST[@]}"; do
        [[ -z "$entry" || "$entry" =~ ^[[:space:]]*# ]] && continue
        n_gpus="${entry%%|*}"
        cmd="${entry#*|}"
        n_gpus=$(echo "$n_gpus" | tr -d ' ')
        cmd=$(echo "$cmd" | sed 's/^[[:space:]]*//')
        TASK_GPUS+=("$n_gpus")
        TASK_CMDS+=("$cmd")
        LAUNCHED+=(0)
        TASK_PIDS+=("")
        TASK_GPU_IDS+=("")
    done
    total=${#TASK_GPUS[@]}

    echo "[table23] Stage '$stage_name': ${total} tasks, GPU cap ${MAX_CONCURRENT_GPUS}, free>=${MIN_VRAM_GB}GB"
    for (( i=0; i<total; i++ )); do
        echo "  task$((i+1)): ${TASK_GPUS[$i]} GPU  →  ${TASK_CMDS[$i]}"
    done

    local ATTEMPT=0 PENDING ACTIVE_GPUS NEED CMD PID NOW ELAPSED
    local -a FREE_IDS ACTIVE_IDS USED_IDS AVAIL SELECTED ASSIGNED_IDS
    while true; do
        ATTEMPT=$((ATTEMPT + 1))
        NOW=$(date +%s)
        ELAPSED=$((NOW - START_TIME))
        if [[ $ELAPSED -gt $MAX_WAIT_SEC ]]; then
            echo "[table23] wait timeout (${MAX_WAIT_HOURS}h)" >&2
            return 1
        fi

        ACTIVE_GPUS=0
        ACTIVE_IDS=()
        for (( i=0; i<total; i++ )); do
            PID=${TASK_PIDS[$i]:-}
            if [[ ${LAUNCHED[$i]} -eq 1 && -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
                ACTIVE_GPUS=$((ACTIVE_GPUS + TASK_GPUS[$i]))
                IFS=',' read -ra ASSIGNED_IDS <<< "${TASK_GPU_IDS[$i]}"
                ACTIVE_IDS+=("${ASSIGNED_IDS[@]}")
            fi
        done

        PENDING=0
        for (( i=0; i<total; i++ )); do
            [[ ${LAUNCHED[$i]} -eq 0 ]] && PENDING=$((PENDING + 1))
        done
        if [[ $PENDING -eq 0 ]]; then
            echo "[table23] Stage '$stage_name': all launched, waiting..."
            wait
            echo "[table23] Stage '$stage_name' done."
            return 0
        fi

        mapfile -t FREE_IDS < <(get_free_gpus)
        echo "[table23] $stage_name check${ATTEMPT} $(gpu_status_str)  free:[${FREE_IDS[*]}]  running:${ACTIVE_GPUS}  pending:${PENDING}"

        USED_IDS=()
        for (( i=0; i<total; i++ )); do
            [[ ${LAUNCHED[$i]} -eq 1 ]] && continue
            NEED=${TASK_GPUS[$i]}
            CMD=${TASK_CMDS[$i]}
            local remaining=$((MAX_CONCURRENT_GPUS - ACTIVE_GPUS - ${#USED_IDS[@]}))
            (( NEED > remaining )) && continue
            AVAIL=()
            local gid used SKIP
            for gid in "${FREE_IDS[@]}"; do
                SKIP=0
                for used in "${ACTIVE_IDS[@]}" "${USED_IDS[@]}"; do
                    [[ "$gid" == "$used" ]] && SKIP=1 && break
                done
                [[ $SKIP -eq 0 ]] && AVAIL+=("$gid")
            done
            if [[ ${#AVAIL[@]} -ge $NEED ]]; then
                SELECTED=("${AVAIL[@]:0:$NEED}")
                USED_IDS+=("${SELECTED[@]}")
                LAUNCHED[$i]=1
                local CUDA_IDS LOG
                CUDA_IDS=$(IFS=','; echo "${SELECTED[*]}")
                LOG="$LOGDIR/table23_${stage_name}_task$((i+1))_$(date +%Y%m%d_%H%M%S).log"
                echo "[table23] launch $stage_name task$((i+1)): CUDA_VISIBLE_DEVICES=${CUDA_IDS} ${CMD}"
                echo "[table23]   log=${LOG}"
                ( eval "CUDA_VISIBLE_DEVICES=${CUDA_IDS} ${CMD}" > "$LOG" 2>&1 ) &
                TASK_PIDS[$i]=$!
                TASK_GPU_IDS[$i]=$CUDA_IDS
            fi
        done
        sleep "$POLL_INTERVAL"
    done
}

case "$STAGE" in
    train)
        load_task_list print-train
        run_stage train
        ;;
    eval)
        load_task_list print-eval
        run_stage eval
        ;;
    collect)
        "$PY" "${PY_ARGS[@]}" collect
        ;;
    all)
        load_task_list print-train
        run_stage train || exit 1
        load_task_list print-eval
        run_stage eval || exit 1
        "$PY" "${PY_ARGS[@]}" collect
        ;;
    names)
        "$PY" "${PY_ARGS[@]}" print-names
        ;;
    *)
        echo "Usage: bash run_table23.sh [all|train|eval|collect|names]" >&2
        exit 1
        ;;
esac
