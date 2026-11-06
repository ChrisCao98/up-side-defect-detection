#!/usr/bin/env bash
# 对一个「分组目录」跑完整链：训练 l2/l3 → 全组打分 → 削库 → 再打分。
#
#   bash runs/run_group.sh products_by_signature/A-E_16/dark AE16dark ae16d
#   bash runs/run_group.sh products_by_signature/A-E_16/dark AE16dark490 ae16d5
#
# 参数：$1=组目录（组内是 <stripe>/<字母>/*.jpg）  $2=数据集名（datasets_<名>）
#       $3=tag 前缀（产出 scores/<tag>_l2_test.csv 等）
#
# 每个 tag 打分的分母是**整组**（含训练过的图），统计误报时按
# scores/<数据集名>_train_list.csv 剔除，见 summarize_ae16.py。
set -u
cd "$(dirname "$0")/.."

GROUP="${1:?用法: run_group.sh <组目录> <数据集名> <tag前缀>}"
NAME="${2:?缺数据集名}"
T="${3:?缺 tag 前缀}"

IMG=up_patchcore-v1
UIDGID="$(id -u):$(id -g)"
DS="datasets_${NAME}"
K=20
# 可由环境变量覆盖；默认值保持与历史实验一致
TRAIN_BS="${TRAIN_BS:-4}"
EVAL_BS="${EVAL_BS:-1}"
WORKERS="${WORKERS:-0}"

[ -d "$GROUP" ] || { echo "✗ 组目录不存在: $GROUP"; exit 1; }
[ -d "$DS/Good/$NAME" ] || { echo "✗ 训练集不存在: $DS/Good/$NAME"; exit 1; }
[ -f "$PWD/weights/pytorch_model.bin" ] || { echo "✗ 缺 weights/pytorch_model.bin"; exit 1; }
echo "组 $GROUP   数据集 $DS   前缀 $T   当前目录 $(pwd)"

train() {  # $1=tag后缀 $2=layer
  local OUT="result_${T}_$1"
  if [ -f "$OUT/Patchcore/F_my/v0/weights/lightning/model.ckpt" ]; then
    echo "[$1] ckpt 已存在，跳过训练"; return 0
  fi
  mkdir -p "$OUT"
  echo "======== 训练 $1 ($2) ========"
  docker run --rm --gpus all -u "$UIDGID" -e MPLCONFIGDIR=/tmp/mpl \
    -v "$PWD/$DS:/app/datasets/testimages_20260824" \
    -v "$PWD/$OUT:/app/result_train" \
    -v "$PWD/weights:/app/weights:ro" \
    -v "$PWD/scripts/train/train_ext.py:/app/train_ext.py" \
    $IMG python train_ext.py --root ./datasets/testimages_20260824 --out ./result_train \
      --train-batch-size "$TRAIN_BS" --eval-batch-size "$EVAL_BS" \
      --num-workers "$WORKERS" --layers "$2" \
      --coreset-ratio 0.10 --image-size 256 --precision 32-true \
    >> "logs/${T}_$1.log" 2>&1 || { echo "[$1] 训练失败，见 logs/${T}_$1.log"; return 1; }
}

score() {  # $1=tag后缀 $2=ckpt 目录
  local OUT="$2"
  [ -f "scores/${T}_$1_test.csv" ] && { echo "[$1] 已有分数，跳过"; return 0; }
  echo "======== 打分 $1 ========"
  docker run --rm --gpus all -u "$UIDGID" -e MPLCONFIGDIR=/tmp/mpl \
    -v "$PWD/scripts/eval/eval_zeromiss.py:/app/eval_zeromiss.py" \
    -v "$PWD/$OUT:/app/r" -v "$PWD/eval_defect149:/app/d" \
    -v "$PWD/$GROUP:/app/g" -v "$PWD/scores:/app/scores" \
    --entrypoint python $IMG /app/eval_zeromiss.py \
      --ckpt /app/r/Patchcore/F_my/v0/weights/lightning/model.ckpt \
      --defect-dir /app/d --good-dir /app/g --tag "${T}_$1" \
      --batch-size "$EVAL_BS" \
    >> "logs/${T}_$1.log" 2>&1 || echo "[$1] 打分失败"
}

prune() {  # $1=tag后缀
  local SRC="result_${T}_$1"
  local PR="result_${T}_$1_pruned"
  local JSON="scores/${T}_$1_prune.json"
  [ -f "scores/${T}_$1_pruned_test.csv" ] && { echo "[$1] 已削过，跳过"; return 0; }
  [ -f "scores/${T}_$1_defect.csv" ] || { echo "[$1] 缺缺陷分数，跳过削库"; return 0; }

  local NAMES
  NAMES=$(python3 - "$T" "$1" <<'PY'
import csv, sys
t, tag = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(f"scores/{t}_{tag}_defect.csv", encoding="utf-8-sig")))
rows.sort(key=lambda r: float(r["raw_score"]))
print(" ".join(r["name"] for r in rows[:7]))
PY
)
  echo "======== 削库 $1 ========"
  mkdir -p "$PR/Patchcore/F_my/v0/weights/lightning"
  docker run --rm --gpus all -u "$UIDGID" -e MPLCONFIGDIR=/tmp/mpl \
    -v "$PWD/scripts/eval/prune_eval.py:/app/prune_eval.py" \
    -v "$PWD/visualize_layers.py:/app/visualize_layers.py" \
    -v "$PWD/$SRC:/app/r" -v "$PWD/eval_defect149:/app/d" \
    -v "$PWD/$GROUP:/app/g" -v "$PWD/scores:/app/scores" \
    --entrypoint python $IMG /app/prune_eval.py \
      --ckpt /app/r/Patchcore/F_my/v0/weights/lightning/model.ckpt \
      --defect-dir /app/d --good-dir /app/g --defect-names $NAMES \
      --k $K --tag "${T}_$1" --save-indices "/app/scores/${T}_$1_prune.json" \
    2>&1 | tail -3

  [ -f "$JSON" ] || { echo "[$1] 没生成 $JSON，跳过"; return 0; }
  # apply_prune 需要 torch，宿主机没有，必须在容器里跑
  docker run --rm -u "$UIDGID" \
    -v "$PWD/scripts/eval/apply_prune.py:/app/apply_prune.py" \
    -v "$PWD/$SRC:/app/s" -v "$PWD/scores:/app/scores" -v "$PWD/$PR:/app/o" \
    --entrypoint python $IMG /app/apply_prune.py \
      --src /app/s/Patchcore/F_my/v0/weights/lightning/model.ckpt \
      --indices "/app/scores/${T}_$1_prune.json" \
      --dst /app/o/Patchcore/F_my/v0/weights/lightning/model.ckpt 2>&1 | tail -2

  score "$1_pruned" "$PR"
}

mkdir -p logs
for pair in "l2 layer2" "l3 layer3"; do
  set -- $pair
  train "$1" "$2" && score "$1" "result_${T}_$1" && prune "$1"
done
echo "GROUP DONE: $T"
