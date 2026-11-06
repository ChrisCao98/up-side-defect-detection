#!/usr/bin/env bash
# 对一个已训练好的分组跑分析链：汇总 → 融合 → 判别头 → 两种留出检验。
#
#   bash runs/analyze_group.sh products_by_signature/A-E_16/dark AE16dark ae16d
#
# 前提：run_group.sh 已经产出 scores/<T>_{l2,l3,l2_pruned,l3_pruned}_{test,defect}.csv
set -u
cd "$(dirname "$0")/.."

GROUP="${1:?用法: analyze_group.sh <组目录> <数据集名> <tag前缀>}"
NAME="${2:?缺数据集名}"
T="${3:?缺 tag 前缀}"

IMG=up_patchcore-v1
UIDGID="$(id -u):$(id -g)"
LIST="scores/${NAME}_train_list.csv"
EXCL="${EXCLUDE:-}"          # 可选：已知缺陷名单，统计误报时剔除
EXCL_HOST=""; EXCL_ARG=""
if [ -n "$EXCL" ]; then EXCL_HOST="--exclude $EXCL"; EXCL_ARG="--exclude /app/$EXCL"; fi
[ -f "$LIST" ] || { echo "✗ 缺训练名单 $LIST"; exit 1; }
mkdir -p logs

echo "======== 1/5 四配置汇总 ========"
python3 scripts/analyze/summarize_ae16.py \
  --tags "${T}_l2" "${T}_l3" "${T}_l2_pruned" "${T}_l3_pruned" \
  --train-list "$LIST" $EXCL_HOST --out "scores/${T}_summary.csv"

echo "======== 2/5 两层融合（扫权重） ========"
python3 scripts/analyze/fuse_ae16.py \
  --a "${T}_l2_pruned" --b "${T}_l3_pruned" --train-list "$LIST" $EXCL_HOST \
  2>&1 | tail -8 | tee "scores/${T}_fusion.txt"

echo "======== 3/5 抽特征 ========"
if [ ! -f "scores/feats_${T}.npz" ]; then
  [ -f "$PWD/weights/pytorch_model.bin" ] || { echo "✗ 缺 weights/pytorch_model.bin"; exit 1; }
  docker run --rm --gpus all -u "$UIDGID" -e MPLCONFIGDIR=/tmp/mpl \
    -v "$PWD/scripts/train/extract_feats.py:/app/extract_feats.py" \
    -v "$PWD/visualize_layers.py:/app/visualize_layers.py" \
    -v "$PWD/weights:/app/weights:ro" \
    -v "$PWD/result_${T}_l3:/app/r" -v "$PWD/eval_defect149:/app/d" \
    -v "$PWD/$GROUP:/app/g" -v "$PWD/scores:/app/scores" \
    --entrypoint python $IMG /app/extract_feats.py \
      --ckpt /app/r/Patchcore/F_my/v0/weights/lightning/model.ckpt \
      --good-dir /app/g --defect-dir /app/d --out "/app/scores/feats_${T}.npz" \
    >> "logs/${T}_feats.log" 2>&1 || { echo "✗ 抽特征失败"; exit 1; }
fi

HEAD_ARGS="--feats /app/scores/feats_${T}.npz $EXCL_ARG \
  --pc-defect /app/scores/${T}_l3_pruned_defect.csv \
  --pc-test /app/scores/${T}_l3_pruned_test.csv \
  --train-list /app/$LIST"

echo "======== 4/5 判别头（0.6/0.4） ========"
docker run --rm --gpus all -u "$UIDGID" -e MPLCONFIGDIR=/tmp/mpl \
  -v "$PWD/scripts/analyze/head_ae16.py:/app/head_ae16.py" \
  -v "$PWD/scores:/app/scores" -v "$PWD/$LIST:/app/$LIST" \
  --entrypoint python $IMG /app/head_ae16.py $HEAD_ARGS \
    --w-pc 0.6 --out-prefix "/app/scores/${T}_head" 2>&1 | tail -5

echo "======== 5/5 留出检验 ========"
for script in head_loo_ae16 head_random_holdout; do
  echo "-- $script"
  docker run --rm -u "$UIDGID" -e MPLCONFIGDIR=/tmp/mpl \
    -v "$PWD/scripts/analyze/$script.py:/app/s.py" \
    -v "$PWD/scores:/app/scores" -v "$PWD/$LIST:/app/$LIST" \
    --entrypoint python $IMG /app/s.py $HEAD_ARGS --w-pc 0.6 \
    2>&1 | tail -4 | tee "scores/${T}_${script}.txt"
done

echo "ANALYZE DONE: $T"
