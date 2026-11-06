# UP-Side Defect Detection

Anomaly detection on grayscale images of the UP side of molded parts. The target is **zero missed defects**: the threshold is the lowest raw score among a 149-image defect library, and every percentage below is the fraction of *unseen* good images that cross it.

[中文](README.md)

## Results

| Dataset | Images | Train | PatchCore only | l2+l3 fusion | + head @1% budget |
|---|---:|---:|---:|---:|---:|
| class_3 (12 subgroups) | 82,478 | 5,170 | 4.36% | 2.68% | **0.95%** |
| class1 / A-D_20 | 4,480 | 720 | 3.67% | 2.71% | **0.88%** |
| class2 / A-E_16 | 2,880 | 200 | 0.60% | **0.30%** | not needed |

- **PatchCore only** = layer3 + bank pruning; **fusion** = l2/l3 each pruned, best image-level weight; **head** = largest PC weight that keeps false positives ≤1%.
- class2 is already under the 1% budget with plain PatchCore, so no head is used.
- The three datasets share no 9-digit prefixes. Reports: [class1](reports/report_class1_AD20.md), [class2](reports/report_class2_AE16.md), [class3](reports/report_class3_all_groups.md) (Chinese).

> All thresholds are borrowed from the 149-image defect library. **Low false positives do not mean this product's defects are caught**: most products in class1/class3 have no defect labels of their own, so recall is only estimated by held-out families.

## Method

The current best pipeline:

1. **Features**: frozen `wide_resnet50_2`, `layer3` only, 256×256 input, FP32. Do not use anomalib's multi-layer concatenation — it resamples to the shallowest grid and destroys layer3's localization.
2. **Memory bank**: one bank per product (or exposure group), coreset ratio 0.10.
3. **Bank pruning**: take the 7 lowest-scoring defects as anchors, delete each anchor's k=20 nearest bank rows, and take the union. Removing ~0.5% of rows cuts false positives 2–5× with no recall cost.
4. **Scoring**: use the raw PatchCore distance from the pruned bank. Do not use the normalized `pred_score`: its constants are calibrated at training time, are not comparable across checkpoints, and clamp scores at 1.0.
5. **Threshold**: the lowest raw score among the 149 defects.
6. **Fusion (optional)**: build and prune l2 and l3 separately, then combine at image level with `w·z_l2 + (1−w)·z_l3`, where z uses the training-good mean/std. Never fuse at patch level.
7. **Discriminative head (optional, needs defect labels)**: layer3 3072-d mean/max/std pooling + logistic regression (C=0.2, balanced), positive = 149 defects, negative = this group's training goods. Fuse as `w·z_pc + (1−w)·z_head`, choosing the largest PC weight with false positives ≤1%.

**Training-set sampling matters just as much:**

- class_3 uses 20 images per product per letter. When products differ in size, keep the total budget fixed and **allocate by product size**: class1 uses 120/480/120 for C57G/C65C/KA1U, which moves false positives from 6.70% to 3.67%.
- Sample **stratified by frame index**. Evenly spaced picks over filename lexicographic order collide with the 20-frame block length and skip whole frame indices; in class1 the first and last frames had no training images and false positives reached 45–97%. `prep_group_dataset.py` defaults to `--sample stratify`.
- Split by brightness only when the **cross-cluster brightness ratio is ≥1.5×**; 1.1–1.2× is not worth splitting.

**Evaluation batch size must be 1.** With batch > 1, images are dropped and scores shift by ~3%; an AUROC of exactly 1.0000 is a red flag.

## Ruled out

Multi-layer concatenation, patch-level fusion, layer4, adding good patches to the bank, flat-field correction, per-image MAD normalization, 3-channel priors, CLAHE, 224 input, merging banks across families, coreset < 0.10, FP16 training. Numbers are in the [class1 report](reports/report_class1_AD20.md) §10 and the [class3 report](reports/report_class3_all_groups.md).

## Layout

```
runs/run_group.sh          train l2/l3 → score → prune → score again
runs/analyze_group.sh      summarize → l2+l3 fusion → features → head → two holdouts
scripts/data/              grouping, brightness split, hardlink tree, training-set sampling
scripts/train/             training, image-level feature extraction
scripts/eval/              raw-score evaluation, prune indices, apply pruning
scripts/analyze/           summary, fusion, head, weight sweep, misses at fixed FP
scripts/report/            Markdown → single-file HTML/PDF
reports/                   the three validation reports
```

## Reproduce

### Environment

Docker + NVIDIA GPU. Image: Python 3.11, torch 2.5.0+cu121, anomalib 2.2.0, lightning 2.5.5, timm 1.0.20.

The host needs `python3` + `numpy` (`summarize_ae16.py` and `fuse_ae16.py` run on the host); report export additionally needs `python-markdown` and `libreoffice`.

```bash
mkdir -p weights
python3 -c "import timm, torch; torch.save(timm.create_model('wide_resnet50_2', pretrained=True).state_dict(), 'weights/pytorch_model.bin')"
docker build -t up_patchcore-v1 .
```

### Data layout

```
/path/to/data/<9-digit prefix>/<letter>/*.jpg   # good images
eval_defect149/**/*.jpg                         # 149 defects, named <prefix>_<letter><index>_UP.jpg
```

### Full pipeline

```bash
# 1 group: stripe (first 4 chars) + per-letter counts → product → signature group
python3 scripts/data/group_products.py --root /path/to/data --out-dir scores

# 2 measure brightness; split only when the cross-cluster ratio is >=1.5x
python3 scripts/data/exposure_groups.py --src /path/to/data
python3 scripts/data/split_by_brightness.py --group A-E_16 \
  --groups scores/product_groups.csv \
  --signatures scores/signature_groups.csv \
  --brightness scores/prefix_brightness.csv \
  --out scores/signature_groups_split.csv

# 3 build the hardlink tree. --base must be on the same filesystem as --src;
#   if --base is outside the repo, symlink it into the repo root.
python3 scripts/data/build_product_layout.py --by-signature --with-stripe \
  --groups scores/signature_groups_split.csv \
  --src /path/to/data --base .

# 4 build the training set (stratify is the default)
python3 scripts/data/prep_group_dataset.py \
  --src products_by_signature/A-E_16/bright --name AE16bright --per-stripe 100

# 5 train l2/l3 → score → prune → score again
EVAL_BS=1 bash runs/run_group.sh products_by_signature/A-E_16/bright AE16bright ae16

# 6 summarize, fuse, head, holdouts
bash runs/analyze_group.sh products_by_signature/A-E_16/bright AE16bright ae16

# 7 head weight sweep at the 1% budget (edit GROUPS at the top of head_weight_sweep.py first)
docker run --rm -u "$(id -u):$(id -g)" -w /app \
  -v "$PWD/scores:/app/scores" \
  -v "$PWD/scripts/analyze/head_weight_sweep.py:/app/h.py" \
  --entrypoint python up_patchcore-v1 /app/h.py --step 0.01
```

**Misses at a fixed false-positive rate** (more stable than the zero-miss threshold, which is pinned by the weakest defects):

```bash
docker run --rm -u "$(id -u):$(id -g)" -w /app \
  -v "$PWD/scores:/app/scores" \
  -v "$PWD/scripts/analyze/miss_at_fp.py:/app/m.py" \
  --entrypoint python up_patchcore-v1 /app/m.py \
  --l2 ae16_l2_pruned --l3 ae16_l3_pruned \
  --train-list scores/AE16bright_train_list.csv \
  --feats scores/feats_ae16.npz --fp 0.01
```

**class1 budget allocation**:

```bash
python3 scripts/data/prep_group_dataset.py --src products_by_signature_class_1/A-D_20 \
  --name class1_AD20r --per-stripe 240 \
  --per-stripe-map "C57G=120,C65C=480,KA1U=120"
```

**Report export** (host needs `python-markdown` and `libreoffice`):

```bash
python3 scripts/report/export_report.py reports/report_class1_AD20.md
```

## Limitations

- The defect library has 149 images from 17 families; most class1/class3 products have no defects of their own, so the threshold is borrowed.
- The zero-miss threshold is pinned by the weakest 2–3 defects. Compare models at a fixed 1% false-positive rate as well.
- class1's remaining false positives sit on C65C's first and last frames: the view scales during the 20-frame burst (width 569→559→569 px). Fix it at capture time; more training images have little marginal value.
- `head_weight_sweep.py` has its group list hard-coded at the top; edit `GROUPS` for a new dataset.
